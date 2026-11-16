"""Deterministic CLAP retrieval for the supplied HW3 MusicCaps reference set.

The required music checkpoint is explicit. No default model, random crop,
caption rewrite, failed-file skipping, or synthetic retrieval fallback is used.
Cache files are local working artifacts and must not enter the submission ZIP.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('HF_HUB_CACHE', str(ROOT / 'cache/huggingface/hub'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'cache/torch'))
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
CLAP_FILE = 'music_audioset_epoch_15_esc_90.14.pt'
CLAP_REPO = 'lukewys/laion_clap'
CLAP_REVISION = 'b3708341862f581175dba5c356a4ebf74a9b6651'
CLAP_SHA256 = 'fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd'
CLAP_BYTES = 2352471003
SAMPLE_RATE = 48000
SAMPLES = 480000
PREPROCESS_POLICY = {
    'schema': 'hw3_clap_audio_v1', 'decoder': 'soundfile float32 native sample rate',
    'channels': 'arithmetic channel mean', 'dc_center': 'subtract mono mean before resampling',
    'resampler': 'torchaudio.functional.resample, float32 CPU',
    'resampling_parameters': {'lowpass_filter_width': 6, 'rolloff': 0.99,
                              'resampling_method': 'sinc_interp_hann', 'beta': None},
    'sample_rate': SAMPLE_RATE, 'frames': SAMPLES,
    'crop': 'first 480000 resampled frames', 'short_audio': 'right zero-pad',
    'quantization': 'exactly once inside official CLAP data API: clip, int16/32767',
    'api_input_length': 'exactly 480000; random crop and repeat-padding paths cannot execute',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Embeddings must be a finite, nonempty 2-D matrix')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError('Zero or invalid embedding norm')
    return values / norms


def rank_topk(text_embeddings, audio_embeddings, reference_ids, k=5):
    """Cosine ranking; exact-score ties use lexical MusicCaps IDs."""
    ids = [str(v) for v in reference_ids]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Reference IDs must be unique and nonempty')
    if not isinstance(k, int) or k < 1 or k > len(ids):
        raise ValueError('Invalid top-k')
    order = sorted(range(len(ids)), key=ids.__getitem__)
    audio = normalize_rows(audio_embeddings)[order]
    text = normalize_rows(text_embeddings)
    if audio.shape[0] != len(ids) or audio.shape[1] != text.shape[1]:
        raise ValueError('Embedding dimensions and reference inventory disagree')
    sorted_ids = [ids[i] for i in order]
    scores = text @ audio.T
    indices = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    return [[{'musiccaps_id': sorted_ids[int(j)], 'cosine': float(scores[i, j])}
             for j in row] for i, row in enumerate(indices)]


def load_captions(path, expected_count=100):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'caption']:
            raise ValueError('Caption CSV must have exactly id,caption columns')
        rows = list(reader)
    ids = [r['id'] for r in rows]
    if len(rows) != expected_count or len(set(ids)) != len(ids):
        raise ValueError('Caption count or ID uniqueness is incorrect')
    if set(ids) != {str(i) for i in range(expected_count)}:
        raise ValueError('IDs must be the exact strings 0 through count-1')
    if any(not r['caption'].strip() for r in rows):
        raise ValueError('Captions cannot be empty')
    return sorted(rows, key=lambda r: int(r['id']))


def validate_prediction(value, reference_ids, expected_count=100):
    if not isinstance(value, dict) or set(value) != {str(i) for i in range(expected_count)}:
        raise ValueError('Retrieval JSON must contain every original caption ID exactly once')
    references = set(reference_ids)
    if any(not isinstance(k, str) or not isinstance(v, str) or v not in references
           for k, v in value.items()):
        raise ValueError('Retrieval keys/values must be strings and values must be actual MusicCaps IDs')


def reference_inventory(directory, expected_count=5367):
    paths = sorted(Path(directory).glob('*.mp3'), key=lambda p: p.stem)
    if len(paths) != expected_count or len({p.stem for p in paths}) != len(paths):
        raise ValueError(f'Expected {expected_count} unique MusicCaps MP3 files')
    rows = []
    for path in paths:
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', path.stem) or path.stat().st_size <= 0:
            raise ValueError(f'Invalid MusicCaps audio file: {path.name}')
        rows.append({'id': path.stem, 'filename': path.name, 'bytes': path.stat().st_size,
                     'sha256': sha256(path)})
    return paths, rows


def preprocess_audio(path):
    """Return (mono float32[480000], audit metadata), before CLAP quantization."""
    import soundfile as sf
    import torch
    import torchaudio

    wave, rate = sf.read(str(path), dtype='float32', always_2d=True)
    if wave.shape[0] < 1 or not np.isfinite(wave).all():
        raise ValueError(f'Empty or nonfinite decoded audio: {Path(path).name}')
    mono = torch.from_numpy(wave).mean(dim=1)
    mono = mono - mono.mean()
    if rate != SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, rate, SAMPLE_RATE,
                                              **PREPROCESS_POLICY['resampling_parameters'])
    resampled_frames = int(mono.numel())
    prepared = np.zeros(SAMPLES, dtype=np.float32)
    count = min(SAMPLES, resampled_frames)
    prepared[:count] = mono[:count].numpy()
    if not np.isfinite(prepared).all():
        raise ValueError('Nonfinite preprocessing result')
    metadata = {'source_sha256': sha256(path), 'source_frames': int(wave.shape[0]),
                'source_rate': int(rate), 'source_channels': int(wave.shape[1]),
                'source_seconds': wave.shape[0] / rate, 'resampled_frames': resampled_frames,
                'cropped_frames': max(0, resampled_frames - SAMPLES),
                'padded_frames': max(0, SAMPLES - resampled_frames),
                'model_input_frames': SAMPLES, 'model_input_rate': SAMPLE_RATE,
                'prepared_peak_absolute': float(np.max(np.abs(prepared))),
                'prepared_rms': float(np.sqrt(np.mean(prepared.astype(np.float64) ** 2))),
                'samples_clipped_by_official_quantization': int(np.count_nonzero(np.abs(prepared) > 1)),
                'preprocess_policy_sha256': object_hash(PREPROCESS_POLICY),
                'prepared_float32_sha256': hashlib.sha256(prepared.tobytes()).hexdigest()}
    return prepared, metadata


def load_clap(checkpoint, device='cpu'):
    """Load only the SHA-verified course music checkpoint with its base architecture."""
    import torch
    import laion_clap
    from transformers import RobertaConfig, RobertaModel
    from unittest.mock import patch

    checkpoint = Path(checkpoint)
    if checkpoint.name != CLAP_FILE or checkpoint.stat().st_size != CLAP_BYTES:
        raise ValueError('Use the exact course-designated CLAP music checkpoint')
    if sha256(checkpoint) != CLAP_SHA256:
        raise ValueError('CLAP checkpoint SHA256 differs from the official pinned HF LFS object')
    if device not in {'cpu', 'mps', 'cuda'}:
        raise ValueError('Specify cpu, mps or cuda explicitly')
    # The complete CLAP object supplies every trained text-encoder parameter.
    # Construct that architecture from its small public config, avoiding a
    # redundant 500MB base-RoBERTa weight download that would be overwritten.
    # The scope ends before inference, and strict checkpoint coverage below is
    # mandatory: no randomly initialized parameter may survive this adapter.
    def configuration_only(name, *args, **kwargs):
        if name != 'roberta-base' or args or kwargs:
            raise ValueError('Unexpected CLAP text encoder factory call; review the pinned library')
        config = RobertaConfig.from_pretrained(name)
        return RobertaModel(config)
    with patch.object(RobertaModel, 'from_pretrained', side_effect=configuration_only):
        module = laion_clap.CLAP_Module(enable_fusion=False, device=device, amodel='HTSAT-base')
    # Official factory transformations, with an explicit modern PyTorch load.
    # Full pickle is restricted here to the exact byte-verified official object.
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint_data.get('state_dict', checkpoint_data)
    if not isinstance(state, dict) or not state:
        raise ValueError('Official checkpoint has no state dictionary')
    if next(iter(state)).startswith('module.'):
        state = {name[7:]: value for name, value in state.items()}
    if 'text_branch.embeddings.position_ids' not in module.model.state_dict():
        state.pop('text_branch.embeddings.position_ids', None)
    module.model.load_state_dict(state, strict=True)
    del state, checkpoint_data
    module.eval()
    return module


def model_provenance(module, device):
    packages = ['torch', 'torchaudio', 'torchvision', 'numpy', 'soundfile', 'laion-clap',
                'transformers', 'timm', 'torchlibrosa']
    versions = {name: importlib.metadata.version(name) for name in packages}
    tokenizer = module.tokenize
    if hasattr(tokenizer, 'backend_tokenizer'):
        tokenizer_spec = tokenizer.backend_tokenizer.to_str()
    else:
        ranks = getattr(tokenizer, 'bpe_ranks', {})
        tokenizer_spec = {'vocabulary': tokenizer.get_vocab(),
                          'merges': [list(pair) for pair, _ in sorted(ranks.items(), key=lambda r: r[1])],
                          'special_tokens': tokenizer.special_tokens_map,
                          'padding_side': tokenizer.padding_side,
                          'truncation_side': tokenizer.truncation_side}
    distribution = importlib.metadata.distribution('laion-clap')
    package_sources = {str(p): sha256(distribution.locate_file(p)) for p in distribution.files
                       if str(p).startswith('laion_clap/') and Path(str(p)).suffix in {'.py', '.json'}}
    return {'checkpoint_filename': CLAP_FILE, 'checkpoint_sha256': CLAP_SHA256,
            'repository': CLAP_REPO, 'revision': CLAP_REVISION,
            'architecture': 'HTSAT-base', 'fusion': False, 'precision': 'float32',
            'device': device, 'versions': versions,
            'tokenizer_fingerprint': object_hash(tokenizer_spec),
            'text_initialization': 'config-only architecture; every trained parameter replaced by strict CLAP checkpoint load',
            'text_model_config_sha256': object_hash(module.model.text_branch.config.to_dict()),
            'text_attention_implementation': getattr(module.model.text_branch.config, '_attn_implementation', None),
            'laion_package_source_sha256': object_hash(package_sources),
            'hook_source_sha256': sha256(inspect.getsourcefile(type(module))),
            'retrieval_source_sha256': sha256(__file__)}


def audio_embeddings(module, waves):
    import torch
    array = np.asarray(waves, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != SAMPLES or not np.isfinite(array).all():
        raise ValueError('Every CLAP input must be one finite, exact 10-second mono waveform')
    with torch.inference_mode():
        result = module.get_audio_embedding_from_data(array, use_tensor=False)
    values = np.asarray(result, dtype=np.float32)
    if values.shape[0] != len(array) or values.ndim != 2:
        raise ValueError('CLAP returned an unexpected audio embedding count')
    normalize_rows(values)
    return values


def text_embeddings(module, captions):
    import torch
    texts = list(captions)
    audit = []
    for caption in texts:
        full = module.tokenize(caption, truncation=False, add_special_tokens=True,
                               return_attention_mask=False)['input_ids']
        audit.append({'caption_sha256': hashlib.sha256(caption.encode()).hexdigest(),
                      'full_token_count_including_special_tokens': len(full),
                      'model_max_tokens': 77, 'was_truncated': len(full) > 77,
                      'dropped_body_tokens': max(0, len(full) - 77),
                      'dropped_suffix_decoded': module.tokenize.decode(full[76:-1]) if len(full) > 77 else ''})
    with torch.inference_mode():
        values = np.asarray(module.get_text_embedding(texts, use_tensor=False), dtype=np.float32)
    if values.shape[0] != len(texts) or values.ndim != 2:
        raise ValueError('CLAP returned an unexpected text embedding count')
    normalize_rows(values)
    return values, audit


class ReferenceCache:
    def __init__(self, directory, context):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.context_hash = object_hash(context)
        self.context = context
        header = self.directory / 'context.json'
        desired = {'schema': 1, 'context_sha256': self.context_hash, 'context': context}
        if header.exists():
            if json.loads(header.read_text()) != desired:
                raise ValueError('Cache context changed; preserve it and choose a distinct cache directory')
        else:
            if any(self.directory.iterdir()):
                raise ValueError('Unowned cache contents without a context manifest; preserve them')
            atomic_json(header, desired)

    def paths(self, reference_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', reference_id):
            raise ValueError('Unsafe reference ID')
        return self.directory / (reference_id + '.npy'), self.directory / (reference_id + '.json')

    def load(self, source):
        array_path, meta_path = self.paths(source['id'])
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        if (meta.get('source') != source or meta.get('context_sha256') != self.context_hash
                or not array_path.is_file() or sha256(array_path) != meta.get('embedding_sha256')):
            raise ValueError('Changed source, context or cached embedding: ' + source['id'])
        array = np.load(array_path, allow_pickle=False)
        if array.ndim != 1:
            raise ValueError('Cached embedding is not a vector')
        normalize_rows(array[None, :])
        return array, meta

    def save(self, source, embedding, audio_audit):
        array_path, meta_path = self.paths(source['id'])
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError('Expected one embedding vector')
        normalize_rows(vector[None, :])
        atomic_array(array_path, vector)
        metadata = {'source': source, 'context_sha256': self.context_hash,
                    'embedding_sha256': sha256(array_path), 'audio': audio_audit}
        atomic_json(meta_path, metadata)
        return metadata


def build_reference(module, paths, inventory, cache, batch_size):
    values, audits, pending = {}, {}, []
    def process_pending():
        if not pending:
            return
        prepared = [preprocess_audio(path) for path, _ in pending]
        result = audio_embeddings(module, [r[0] for r in prepared])
        for (path, source), vector, (_, meta) in zip(pending, result, prepared):
            if meta['source_sha256'] != source['sha256']:
                raise ValueError('Source audio changed while embedding: ' + source['id'])
            audits[source['id']] = cache.save(source, vector, meta)
            values[source['id']] = vector
        pending.clear()
        print(json.dumps({'reference_embeddings_complete': len(values), 'expected': len(paths)}), flush=True)
    for path, source in zip(paths, inventory):
        cached = cache.load(source)
        if cached is not None:
            values[source['id']], audits[source['id']] = cached
        else:
            pending.append((path, source))
            if len(pending) == batch_size:
                process_pending()
    process_pending()
    matrix = np.stack([values[row['id']] for row in inventory])
    normalize_rows(matrix)
    return matrix, [audits[row['id']] for row in inventory]


def check_frozen_protocol(path, captions_path=None, inventory=None, device=None, batch_size=None):
    value = json.loads(Path(path).read_text())
    if value.get('status') != 'frozen':
        raise ValueError('Formal retrieval requires a frozen protocol, not a draft')
    clap = value.get('retrieval', value.get('evaluation', {}).get('clap', {}))
    if clap.get('checkpoint_sha256') != CLAP_SHA256:
        raise ValueError('Frozen protocol does not identify the required CLAP checkpoint SHA')
    if clap.get('preprocess_policy_sha256') != object_hash(PREPROCESS_POLICY):
        raise ValueError('Frozen protocol must match the exact preprocessing-policy SHA')
    if clap.get('retrieval_source_sha256') != sha256(__file__):
        raise ValueError('Retrieval implementation changed after protocol freezing')
    if captions_path is not None and clap.get('captions_sha256') != sha256(captions_path):
        raise ValueError('Test captions changed after protocol freezing')
    if inventory is not None and clap.get('reference_inventory_sha256') != object_hash(inventory):
        raise ValueError('MusicCaps inventory changed after protocol freezing')
    if device is not None and clap.get('device') != device:
        raise ValueError('Retrieval device differs from the frozen runtime')
    if batch_size is not None and clap.get('batch_size') != batch_size:
        raise ValueError('Retrieval batch size differs from the frozen runtime')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--audio-dir', type=Path, default=ROOT / 'audio_mp3')
    parser.add_argument('--captions', type=Path, default=ROOT / 'sample_code/test.csv')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'cache/clap_reference/main')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/retrieval')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--expected-reference-count', type=int, default=5367)
    parser.add_argument('--mode', choices=['reference', 'retrieve', 'smoke'], default='retrieve')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--smoke-count', type=int, default=5)
    parser.add_argument('--smoke-prompt', default='A dry solo piano playing separated notes, without speech.')
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError('Batch size must be positive')
    if args.mode == 'retrieve':
        if args.protocol is None:
            raise ValueError('Specify the frozen retrieval protocol before using test captions')
        check_frozen_protocol(args.protocol, device=args.device, batch_size=args.batch_size)
    started = time.monotonic()
    paths, inventory = reference_inventory(args.audio_dir, args.expected_reference_count)
    if args.mode == 'retrieve':
        check_frozen_protocol(args.protocol, args.captions, inventory)
    if args.mode == 'smoke':
        if not 1 <= args.smoke_count <= len(paths):
            raise ValueError('Invalid non-test smoke size')
        paths, inventory = paths[:args.smoke_count], inventory[:args.smoke_count]
        args.cache_dir = args.cache_dir.parent / (args.cache_dir.name + '_non_test_smoke')
        args.output_dir = args.output_dir / 'non_test_smoke'
    module = load_clap(args.checkpoint, args.device)
    provenance = model_provenance(module, args.device)
    context = {'model': provenance, 'preprocess': PREPROCESS_POLICY, 'batch_size': args.batch_size,
               'reference_inventory': inventory}
    cache = ReferenceCache(args.cache_dir, context)
    audio, records = build_reference(module, paths, inventory, cache, args.batch_size)
    reference_ids = [row['id'] for row in inventory]
    manifest = {'status': 'reference_embeddings_complete', 'reference_count': len(paths),
                'mode': args.mode, 'batch_size': args.batch_size,
                'cache_context_sha256': cache.context_hash, 'model': provenance,
                'preprocessing': PREPROCESS_POLICY, 'records': records,
                'reference_embedding_shape': list(audio.shape), 'human_listening_performed': False}
    if args.mode != 'reference':
        rows = ([{'id': 'smoke', 'caption': args.smoke_prompt}] if args.mode == 'smoke'
                else load_captions(args.captions))
        text, token_audit = text_embeddings(module, [r['caption'] for r in rows])
        ranking = rank_topk(text, audio, reference_ids, min(5, len(reference_ids)))
        details = {r['id']: {'original_caption': r['caption'], 'tokenizer': t,
                             'topk': scores} for r, t, scores in zip(rows, token_audit, ranking)}
        manifest.update(status='non_test_smoke_complete' if args.mode == 'smoke' else 'retrieval_complete',
                        caption_count=len(rows), text_embeddings_shape=list(text.shape),
                        captions_truncated=sum(r['was_truncated'] for r in token_audit),
                        tie_policy='exact float64 cosine score ties use lexically sorted MusicCaps IDs',
                        scoring='L2 normalization and cosine in float64 from float32 model embeddings',
                        caption_source_sha256=sha256(args.captions) if args.mode == 'retrieve' else None)
        atomic_json(args.output_dir / 'topk_and_tokenization.json', details)
        if args.mode == 'retrieve':
            prediction = {r['id']: scores[0]['musiccaps_id'] for r, scores in zip(rows, ranking)}
            validate_prediction(prediction, reference_ids)
            manifest['protocol_sha256'] = sha256(args.protocol)
            manifest['unique_retrieved_references'] = len(set(prediction.values()))
            atomic_json(args.output_dir / 'retrieval.json', prediction)
            manifest['retrieval_json_sha256'] = sha256(args.output_dir / 'retrieval.json')
    manifest['elapsed_seconds'] = time.monotonic() - started
    atomic_json(args.output_dir / 'retrieval_manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('status', 'reference_count', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
