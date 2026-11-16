"""Audio-only adapter for the required, pinned av-benchmark metrics.

The upstream VGGish, PaSST and metric implementations are retained. This adapter
avoids unrelated video models, supports MP3 input and rejects missing pairs.
Reference selection must be stated explicitly; a retrieved clip is a proxy,
not ground-truth audio for the new caption.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AV_COMMIT = 'f351b9a6fc6abde746d5f8e1d4c47c883319cb41'
PASST_COMMIT = '1ab778984f67cca09dd8bbad11f728a0122cce24'
AV = ROOT / 'vendor/av-benchmark'
PASST = ROOT / 'vendor/passt_hear21'
VGG_URL = 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth'
POLICY = {'mono': 'channel mean', 'center': 'subtract mono mean before resampling',
          'resample': 'torchaudio sinc_interp_hann, lowpass_width6, rolloff.99',
          'duration': 'first 10s', 'vggish': '16kHz, no right padding, upstream log-mel and frame encoder',
          'passt': '32kHz, right zero-pad to 320000 frames as upstream',
          'source': 'decode the actual MP3 files with soundfile float32'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def upstream():
    """Use the reviewed source files without importing the video extractor."""
    if not (AV / 'av_bench/metrics/fad.py').is_file():
        raise ValueError('The pinned av-benchmark source is required in vendor/av-benchmark')
    sys.path.insert(0, str(AV))
    from av_bench.metrics import compute_fd, compute_kl
    return compute_fd, compute_kl


def prepared_wave(path, rate, pad):
    import soundfile as sf
    import torch
    import torchaudio
    x, original_rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError(f'Invalid decoded audio: {path}')
    wave = torch.from_numpy(x).mean(dim=1)
    wave = wave - wave.mean()
    if original_rate != rate:
        wave = torchaudio.functional.resample(wave, original_rate, rate,
                     lowpass_filter_width=6, rolloff=.99, resampling_method='sinc_interp_hann')
    wave = wave[:rate * 10]
    if len(wave) < int(rate * .96):
        raise ValueError('Audio is too short for the required VGGish evaluation')
    if pad and len(wave) < rate * 10:
        wave = torch.nn.functional.pad(wave, (0, rate * 10 - len(wave)))
    return wave, {'source_rate': original_rate, 'source_frames': len(x),
                  'source_channels': x.shape[1], 'prepared_rate': rate,
                  'prepared_frames': len(wave),
                  'prepared_float32_sha256': hashlib.sha256(wave.numpy().tobytes()).hexdigest()}


def load_encoder(kind, device):
    import torch
    from tools.prepare_evaluation_models import ensure
    upstream()
    torch.hub.set_dir(str(ROOT / 'cache/torch/hub'))
    # Publish no checkpoint until its complete source size and full pinned
    # mirror SHA match. Corrupt prior Torch Hub downloads are quarantined.
    weight, transport = ensure(kind, download_missing=True)
    if kind == 'vggish':
        from av_bench.vggish.vggish import VGGish
        model = VGGish(device=device, pretrained=False, postprocess=False)
        state = torch.load(weight, map_location='cpu', weights_only=True)
        ignored = sorted(k for k in state if k.startswith('pproc.'))
        # The benchmark disables the optional postprocessor; every actual
        # feature-encoder parameter must nevertheless match strictly.
        model.load_state_dict({k: v for k, v in state.items() if k not in ignored}, strict=True)
        provenance = {'url': VGG_URL, 'sha256': digest(weight),
                      'unused_postprocessor_checkpoint_keys': ignored,
                      'source_sha256': digest(AV / 'av_bench/vggish/vggish.py'),
                      'frontend_sha256': digest(AV / 'av_bench/vggish/mel_features.py')}
    else:
        sys.path.insert(0, str(PASST))
        from hear21passt.base import get_basic_model
        from hear21passt.models.helpers import vit_helpers
        from unittest.mock import patch
        def verified_local_weights(url, **kwargs):
            if url != transport['original_release']['url']:
                raise ValueError('The upstream PaSST factory requested another checkpoint')
            return str(weight)
        # Retain the official architecture, checkpoint filter and strict
        # parameter loading while making the transport strictly local.
        with patch.object(vit_helpers, 'download_cached_file', side_effect=verified_local_weights):
            model = get_basic_model(mode='all')
        provenance = {'model': 'passt_s_swa_p16_128_ap476', 'source_commit': PASST_COMMIT,
                      'weights_sha256': digest(weight),
                      'weight_loading': 'upstream factory/filter/strict load; transport returns the preverified local checkpoint',
                      'source_sha256': digest(PASST / 'hear21passt/models/passt.py'),
                      'frontend_sha256': digest(PASST / 'hear21passt/models/preprocess.py')}
    model = model.to(device).eval()
    provenance.update(device=device, dtype='float32', av_benchmark_commit=AV_COMMIT,
                      policy=POLICY, adapter_sha256=digest(__file__),
                      helper_source_sha256=transport['helper_source_sha256'],
                      weight_transport=transport,
                      versions={p: importlib.metadata.version(p) for p in
                                ['torch', 'torchaudio', 'numpy', 'scipy', 'soundfile']})
    return model, provenance


def extract(args):
    import torch
    directory = args.audio_dir.resolve()
    files = sorted([p for p in directory.iterdir() if p.suffix.lower() in {'.wav', '.flac', '.mp3'}],
                   key=lambda p: p.stem)
    if not files or len({p.stem for p in files}) != len(files):
        raise ValueError('Require nonempty audio with unique stems; no failed-file skipping is supported')
    if args.ids:
        requested = json.loads(args.ids.read_text())
        if isinstance(requested, dict): requested = list(requested.values())
        requested = set(requested)
        if not requested.issubset({p.stem for p in files}):
            raise ValueError('An explicitly requested reference ID is missing')
        files = [p for p in files if p.stem in requested]
    model, provenance = load_encoder(args.kind, args.device)
    context = {'kind': args.kind, 'provenance': provenance,
               'inventory': [{'id': p.stem, 'sha256': digest(p)} for p in files]}
    fingerprint = object_hash(context)
    target = args.cache.resolve(); target.mkdir(parents=True, exist_ok=True)
    index = target / 'index.json'
    previous_records = {}
    if index.exists():
        previous = json.loads(index.read_text())
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Cache model, source audio, policy or runtime changed; preserve it and use another cache directory')
        previous_records = {r['id']: r for r in previous['records']}
    records = []
    for i, path in enumerate(files):
        artifact = target / f'{path.stem}.npz'
        if artifact.exists():
            if path.stem in previous_records and digest(artifact) != previous_records[path.stem]['sha256']:
                raise ValueError('Feature bytes changed after recording; do not bless them by resuming')
            with np.load(artifact, allow_pickle=False) as z:
                values = z['features']
                if str(z['fingerprint']) != fingerprint or not np.isfinite(values).all():
                    raise ValueError('Corrupt or stale feature cache')
        else:
            wave, meta = prepared_wave(path, 16000 if args.kind == 'vggish' else 32000,
                                       pad=args.kind == 'passt')
            with torch.inference_mode():
                result = model(wave.unsqueeze(0) if args.kind == 'vggish' else wave.unsqueeze(0).to(args.device))
            values = result[0].detach().float().cpu().numpy()
            if args.kind == 'passt': values = values[:527]
            if (args.kind == 'vggish' and (values.ndim != 2 or values.shape[1] != 128)) or \
               (args.kind == 'passt' and values.shape != (527,)) or not np.isfinite(values).all():
                raise ValueError(f'Unexpected {args.kind} output shape/value: {values.shape}')
            temporary = artifact.with_suffix('.npz.tmp')
            with temporary.open('wb') as f:
                np.savez(f, features=values, fingerprint=np.array(fingerprint),
                         preprocessing_json=np.array(json.dumps(meta)))
            temporary.replace(artifact)
        records.append({'id': path.stem, 'source_sha256': context['inventory'][i]['sha256'],
                        'file': artifact.name, 'sha256': digest(artifact), 'shape': list(values.shape)})
        write_json(index, {'completed': len(records) == len(files), 'fingerprint': fingerprint,
                          **context, 'completed_count': len(records), 'records': records})
        print(json.dumps({'kind': args.kind, 'completed': len(records), 'total': len(files)}), flush=True)


def read_cache(directory, kind):
    index = json.loads((directory / 'index.json').read_text())
    if not index['completed'] or index['kind'] != kind:
        raise ValueError('An entire feature cache with the correct encoder is required')
    values = {}
    for row in index['records']:
        path = directory / row['file']
        if digest(path) != row['sha256']: raise ValueError('Cached features changed')
        with np.load(path, allow_pickle=False) as z:
            if str(z['fingerprint']) != index['fingerprint']: raise ValueError('Cache fingerprint mismatch')
            values[row['id']] = z['features'].copy()
    if len(values) != index['completed_count']: raise ValueError('Duplicate cache ID')
    return values, index


def aligned_metrics(reference_vgg, generated_vgg, reference_logits, generated_logits,
                    retrieval, captions, text_embeddings, audio_embeddings):
    """Exact upstream FD/KL plus per-caption cosine; fail closed on coverage."""
    import torch
    from torch.nn import functional as F
    compute_fd, compute_kl = upstream()
    ids = [str(i) for i in range(100)]
    if any(set(d) != set(ids) for d in [generated_vgg, generated_logits, retrieval, captions]):
        raise ValueError('Every generated/pair/caption ID must be present exactly once (0..99)')
    if any(not isinstance(v, str) or v not in reference_vgg or v not in reference_logits for v in retrieval.values()):
        raise ValueError('Every retrieved proxy must have both reference feature types')
    pred = np.concatenate([generated_vgg[i] for i in ids], axis=0)
    full_ref = np.concatenate([reference_vgg[i] for i in sorted(reference_vgg)], axis=0)
    paired_ref = np.concatenate([reference_vgg[retrieval[i]] for i in ids], axis=0)
    g = torch.as_tensor(np.stack([generated_logits[i] for i in ids]), dtype=torch.float32)
    r = torch.as_tensor(np.stack([reference_logits[retrieval[i]] for i in ids]), dtype=torch.float32)
    if g.shape != (100, 527) or r.shape != (100, 527) or not torch.isfinite(g).all() or not torch.isfinite(r).all():
        raise ValueError('PaSST must produce 100 finite 527-class vectors')
    kl = compute_kl([g], r)['kl_softmax']
    per_kl = F.kl_div(F.log_softmax(g, dim=1), F.log_softmax(r, dim=1),
                      reduction='none', log_target=True).sum(dim=1)
    if abs(float(per_kl.mean()) - kl) > 1e-5: raise ValueError('Paired KL disagrees with upstream')
    text = torch.as_tensor(text_embeddings, dtype=torch.float32)
    audio = torch.as_tensor(audio_embeddings, dtype=torch.float32)
    if text.ndim != 2 or text.shape != audio.shape or text.shape[0] != 100:
        raise ValueError('CLAP requires 100 aligned text/audio vectors')
    if not torch.isfinite(text).all() or not torch.isfinite(audio).all() or (text.norm(dim=1) <= 0).any() or (audio.norm(dim=1) <= 0).any():
        raise ValueError('Invalid CLAP embeddings')
    per_clap = torch.cosine_similarity(text, audio, dim=-1)
    shifted = torch.cosine_similarity(torch.roll(text, shifts=1, dims=0), audio, dim=-1)
    fd_all = float(compute_fd(pred, full_ref)); fd_pair = float(compute_fd(pred, paired_ref))
    if not all(np.isfinite(v) and v >= -1e-6 for v in [fd_all, fd_pair, kl]):
        raise ValueError('The upstream metric failed; a sentinel/NaN is not a score')
    return {'FD-VGG-all-reference': fd_all, 'FD-VGG-retrieved-proxy': fd_pair,
            'KL-PASST-softmax-retrieved-proxy': float(kl), 'CLAP': float(per_clap.mean()),
            'counts': {'generated_clips': 100, 'fd_reference_clips': len(reference_vgg),
                       'paired_reference_rows': 100, 'unique_retrieved_references': len(set(retrieval.values()))},
            'protocol_status': 'Reference policy is explicitly reported; TA interpretation remains unconfirmed.',
            'reference_is_ground_truth': False,
            'clap_pairing_control': {
                'policy': 'Pair generated ID i with original caption (i-1) modulo 100; fixed before test scoring.',
                'matched_mean': float(per_clap.mean()), 'cyclic_mismatch_mean': float(shifted.mean()),
                'matched_minus_mismatch_mean': float((per_clap - shifted).mean()),
                'matched_greater_fraction': float((per_clap > shifted).float().mean()),
                'interpretation': 'A caption sensitivity diagnostic; shifted music captions can still share concepts. It is not a human quality score.'},
            'per_caption': [{'id': i, 'reference_id': retrieval[i], 'clap': float(per_clap[j]),
                             'clap_cyclic_mismatch': float(shifted[j]),
                             'kl_passt_softmax': float(per_kl[j])} for j, i in enumerate(ids)]}


def score(args):
    from retrieve import load_captions, load_clap, model_provenance, text_embeddings, audio_embeddings, preprocess_audio
    rows = load_captions(args.captions)
    captions = {r['id']: r['caption'] for r in rows}
    retrieval = json.loads(args.retrieval.read_text())
    rv, rvi = read_cache(args.reference_vgg, 'vggish'); gv, gvi = read_cache(args.generated_vgg, 'vggish')
    rl, rli = read_cache(args.reference_passt, 'passt'); gl, gli = read_cache(args.generated_passt, 'passt')
    if len(rv) != args.expected_reference_count:
        raise ValueError('The FD reference population is incomplete or differs from its declared count')
    ref_hashes = {r['id']: r['source_sha256'] for r in rvi['records']}
    if any(ref_hashes.get(r['id']) != r['source_sha256'] for r in rli['records']):
        raise ValueError('Reference VGGish and PaSST features came from different source audio')
    expected_ids = {str(i) for i in range(100)}
    if set(gv) != expected_ids or set(gl) != expected_ids:
        raise ValueError('All 100 generated clips must be encoded by each required metric')
    generated_hashes = {r['id']: r['source_sha256'] for r in gvi['records']}
    if any(generated_hashes.get(r['id']) != r['source_sha256'] for r in gli['records']):
        raise ValueError('Generated VGGish and PaSST features came from different source audio')
    for i in range(100):
        if digest(args.generated_audio / f'{i}.mp3') != generated_hashes[str(i)]:
            raise ValueError('Actual submitted MP3 changed after feature extraction')
    # The only allowed extractor difference is which source files are present.
    if rvi['provenance'] != gvi['provenance'] or rli['provenance'] != gli['provenance']:
        raise ValueError('Reference and generated encoders/runtime/preprocessing disagree')
    module = load_clap(args.clap_checkpoint, args.device)
    te, audit = text_embeddings(module, [captions[str(i)] for i in range(100)])
    ae, input_records = [], []
    for i in range(100):
        path = args.generated_audio / f'{i}.mp3'
        wave, meta = preprocess_audio(path)
        ae.append(audio_embeddings(module, [wave])[0]); input_records.append({'id': str(i), **meta})
    result = aligned_metrics(rv, gv, rl, gl, retrieval, captions, te, np.stack(ae))
    silence = audio_embeddings(module, [np.zeros(480000, dtype=np.float32)])[0]
    silence_scores = np.sum((te / np.linalg.norm(te, axis=1, keepdims=True))
                            * (silence / np.linalg.norm(silence)), axis=1)
    matched = np.array([row['clap'] for row in result['per_caption']])
    result['clap_silence_control'] = {
        'policy': 'One exact 10-second zero waveform, same required checkpoint and CLAP API.',
        'silence_mean': float(silence_scores.mean()),
        'generated_minus_silence_mean': float((matched - silence_scores).mean()),
        'generated_greater_fraction': float(np.mean(matched > silence_scores)),
        'interpretation': 'A negative control of CLAP only. A higher generated score does not establish musical quality; this uses one silence control, not a negative-audio population.'}
    for row, value in zip(result['per_caption'], silence_scores):
        row['clap_silence'] = float(value)
    result.update(completed=True, generated_audio_format='actual submitted MP3',
                  av_benchmark_commit=AV_COMMIT, adapter_sha256=digest(__file__),
                  caption_csv_sha256=digest(args.captions), retrieval_sha256=digest(args.retrieval),
                  clap_provenance=model_provenance(module, args.device), tokenizer_audit=audit,
                  clap_audio_inputs=input_records,
                  feature_cache_indices={k: digest(v / 'index.json') for k, v in
                      [('reference_vgg', args.reference_vgg), ('generated_vgg', args.generated_vgg),
                       ('reference_passt', args.reference_passt), ('generated_passt', args.generated_passt)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vectors = args.output.with_suffix('.clap_features.npz')
    temporary = vectors.with_suffix(vectors.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, text=te, audio=np.stack(ae), silence=silence,
                 ids=np.array([str(i) for i in range(100)]))
    temporary.replace(vectors)
    result['clap_feature_archive'] = {'filename': vectors.name, 'sha256': digest(vectors)}
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if isinstance(v, (float, bool))}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--kind', choices=['vggish', 'passt'], required=True)
    p.add_argument('--audio-dir', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--ids', type=Path, help='JSON ID list or retrieval mapping; reference extraction only')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p = sub.add_parser('score')
    p.add_argument('--captions', type=Path, default=ROOT / 'sample_code/test.csv')
    p.add_argument('--retrieval', type=Path, required=True)
    p.add_argument('--expected-reference-count', type=int, default=5367)
    for name in ['reference-vgg', 'generated-vgg', 'reference-passt', 'generated-passt', 'generated-audio', 'clap-checkpoint', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    args = parser.parse_args()
    (extract if args.command == 'extract' else score)(args)


if __name__ == '__main__':
    main()
