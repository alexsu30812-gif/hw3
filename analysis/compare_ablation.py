"""Paired, predeclared CFG ablation on a fixed caption cohort.

This is a subset analysis, never a replacement for the required 100-clip scores.
CLAP is computed from the actual MP3 files; KL uses declared retrieved proxies.
No result is written unless all four conditions cover the entire frozen cohort.

Example (repeat --condition for all four frozen conditions):
  python analysis/compare_ablation.py --mode run --protocol protocol.json \
    --condition baseline=results/baseline,cache/base_vgg,cache/base_passt \
    --reference-vgg cache/reference_vgg --reference-passt cache/reference_passt \
    --retrieval results/retrieval/retrieval.json --clap-checkpoint cache/models/music_audioset_epoch_15_esc_90.14.pt \
    --output-dir results/ablation --device cpu

Modes extract/compare allow expensive CLAP inference to be cached separately.
All required tokenizer/config files must already be available locally.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import generate as generation
import evaluate as evaluation
import retrieve

FACTOR_PAIRS = {(False, False), (True, False), (False, True), (True, True)}
VARYING_SETTINGS = {'guidance_scale', 'end_scale', 'schedule', 'rescale'}
METRICS = ('CLAP', 'KL-PASST-softmax-retrieved-proxy', 'rms', 'peak', 'fraction_abs_ge_1')


def read_json(path):
    return json.loads(Path(path).read_text())


def frozen_design(protocol):
    if protocol.get('status') != 'frozen' or protocol.get('no_test_or_validation_tuning') is not True:
        raise ValueError('A frozen no-test-tuning protocol is required')
    design = protocol['ablation']
    ids = design['cohort_ids']
    if (not isinstance(ids, list) or len(ids) < 2 or len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or i not in {str(j) for j in range(100)} for i in ids)):
        raise ValueError('Predeclared cohort must contain unique original caption IDs')
    conditions = design['conditions']
    if len(conditions) != 4 or len({c['id'] for c in conditions}) != 4:
        raise ValueError('Exactly four unique factorial conditions are required')
    if any(type(c.get(k)) is not bool for c in conditions for k in ('schedule', 'rescale')):
        raise ValueError('Factor values must be explicit booleans')
    if {(c['schedule'], c['rescale']) for c in conditions} != FACTOR_PAIRS:
        raise ValueError('The complete 2x2 factorial design is required')
    seed, draws = design['bootstrap']['seed'], design['bootstrap']['resamples']
    if type(seed) is not int or seed < 0 or type(draws) is not int or not 1000 <= draws <= 100000:
        raise ValueError('Specify a nonnegative bootstrap seed and 1000..100000 draws before evaluation')
    return ids, conditions, seed, draws


def validate_group(protocol_path, captions_path, directories):
    """Check formal manifests, exact frozen settings and paired initial noise."""
    protocol = read_json(protocol_path)
    ids, factors, _, _ = frozen_design(protocol)
    captions = {r['id']: r['caption'] for r in generation.read_captions(captions_path)}
    if set(directories) != {c['id'] for c in factors}:
        raise ValueError('Every frozen condition must be supplied exactly once')
    manifests, records, settings = {}, {}, {}
    invariant, schedule_levels, rescale_levels, noise = None, {}, {}, {}
    actual_sources = {'generate.py': generation.digest(ROOT / 'generate.py'),
                      'cfg.py': generation.digest(ROOT / 'cfg.py')}
    for factor in factors:
        name = factor['id']; directory = Path(directories[name])
        manifest = read_json(directory / 'manifest.json'); identity = manifest['identity']
        setting = identity['settings']
        checked = generation.validate_frozen_protocol(protocol_path, captions_path, setting)
        if (checked['variant'] != name or identity.get('protocol') != checked
                or identity.get('mode') != 'formal_test'
                or identity.get('csv_sha256') != generation.digest(captions_path)
                or identity.get('source_sha256') != actual_sources
                or manifest['run_sha256'] != generation.text_sha(generation.canonical(identity))):
            raise ValueError('Manifest is not the exact formal run of its frozen condition: ' + name)
        if not identity.get('model_files') or not identity.get('runtime_versions') or not identity.get('codecs'):
            raise ValueError('Actual model/runtime/codec provenance is required')
        constant = {k: v for k, v in setting.items() if k not in VARYING_SETTINGS}
        control = {k: identity[k] for k in ('model_files', 'runtime_versions', 'source_sha256', 'codecs', 'prompts')}
        control['settings'] = constant
        if invariant is None: invariant = control
        if control != invariant:
            raise ValueError('A nonexperimental model, runtime, caption, codec or setting changed: ' + name)
        if setting['schedule'] != ('linear' if factor['schedule'] else 'constant'):
            raise ValueError('Schedule factor disagrees with actual generation settings')
        if (setting['rescale'] > 0) != factor['rescale']:
            raise ValueError('Rescaling factor disagrees with actual generation settings')
        schedule_key = {k: setting[k] for k in ('schedule', 'guidance_scale', 'end_scale')}
        level = factor['schedule']
        if level in schedule_levels and schedule_levels[level] != schedule_key:
            raise ValueError('Schedule settings changed across the other factor')
        schedule_levels[level] = schedule_key
        level = factor['rescale']
        if level in rescale_levels and rescale_levels[level] != setting['rescale']:
            raise ValueError('Rescaling strength changed across the other factor')
        rescale_levels[level] = setting['rescale']
        samples = {s['id']: s for s in manifest['samples']}
        if len(samples) != len(manifest['samples']) or not set(ids).issubset(samples):
            raise ValueError('Missing or duplicate predeclared cohort IDs: ' + name)
        for identifier in ids:
            row = samples[identifier]
            caption_sha = generation.text_sha(captions[identifier])
            contract = generation.text_sha(generation.canonical({'run_sha256': manifest['run_sha256'],
                        'id': identifier, 'caption_sha256': caption_sha}))
            if row.get('caption_sha256') != caption_sha or row.get('seed') != setting['base_seed'] + int(identifier):
                raise ValueError('Caption/seed pairing changed for ' + identifier)
            generation.verify_record(directory, row, contract)
            sampling = row['sampling']
            paired = {'seed': row['seed'], 'noise': sampling['initial_latents_sha256_float32'],
                      'latent_shape': sampling['latent_shape'], 'scheduler': sampling['scheduler']}
            if not isinstance(paired['noise'], str) or len(paired['noise']) != 64:
                raise ValueError('Missing actual initial-noise hash')
            if identifier in noise and noise[identifier] != paired:
                raise ValueError('Initial noise, seed, latent shape or scheduler differs across conditions: ' + identifier)
            noise[identifier] = paired
        manifests[name], records[name], settings[name] = manifest, samples, setting
    return protocol, ids, factors, captions, manifests, records, settings


def bootstrap_mean(values, seed, resamples):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires at least two finite paired caption values')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {'mean': float(values.mean()), 'percentile_95_ci': [float(low), float(high)],
            'n_captions': len(values), 'seed': seed, 'resamples': resamples}


def factorial_effects(values, factors):
    """Effects are per caption, retaining pairing for bootstrap intervals."""
    cells = {(c['schedule'], c['rescale']): np.asarray(values[c['id']], dtype=np.float64) for c in factors}
    if len({v.shape for v in cells.values()}) != 1 or any(v.ndim != 1 or not np.isfinite(v).all() for v in cells.values()):
        raise ValueError('Factorial cells must be finite aligned caption vectors')
    a, b, c, d = cells[False, False], cells[True, False], cells[False, True], cells[True, True]
    return {'schedule_main_effect': ((b - a) + (d - c)) / 2,
            'rescale_main_effect': ((c - a) + (d - b)) / 2,
            'interaction_difference_in_differences': d - b - c + a}


def context_for(args, ids, factors, records):
    return {'protocol_sha256': generation.digest(args.protocol), 'captions_sha256': generation.digest(args.captions),
            'retrieval_sha256': generation.digest(args.retrieval), 'cohort_ids': ids,
            'clap_checkpoint_sha256': retrieve.CLAP_SHA256,
            'clap_preprocess_policy_sha256': retrieve.object_hash(retrieve.PREPROCESS_POLICY),
            'retrieval_source_sha256': generation.digest(ROOT / 'retrieve.py'),
            'clap_versions': {p: importlib.metadata.version(p) for p in
                              ('torch', 'torchaudio', 'laion-clap', 'transformers', 'numpy', 'soundfile')},
            'conditions': {c['id']: {'manifest_sha256': generation.digest(args.conditions[c['id']][0] / 'manifest.json'),
                'mp3_inputs': [{'id': i, 'sha256': records[c['id']][i]['mp3']['sha256']} for i in ids]}
                for c in factors}}


def extract_clap(args, context, ids, factors, captions, records):
    target = args.output_dir / 'clap_features.npz'; index = args.output_dir / 'clap_manifest.json'
    if index.exists() or target.exists():
        return read_clap(args.output_dir, context, ids, factors)
    # A complete local cache is required; no hidden pretrained-model download.
    os.environ['HF_HUB_OFFLINE'] = '1'
    module = retrieve.load_clap(args.clap_checkpoint, args.device)
    text, token_audit = retrieve.text_embeddings(module, [captions[i] for i in ids])
    features, signal = {}, {}
    import soundfile as sf
    for factor in factors:
        name = factor['id']; arrays, audits = [], []
        directory = args.conditions[name][0]
        for identifier in ids:
            path = directory / records[name][identifier]['mp3']['path']
            wave, metadata = retrieve.preprocess_audio(path)
            arrays.append(retrieve.audio_embeddings(module, [wave])[0])
            actual, rate = sf.read(path, dtype='float32', always_2d=True)
            if actual.ndim != 2 or actual.shape[1] != 2 or rate != 44100 or not np.isfinite(actual).all():
                raise ValueError('Expected genuine finite stereo 44.1kHz generated MP3')
            audits.append({'id': identifier, **metadata,
                'rms': float(np.sqrt(np.mean(actual.astype(np.float64) ** 2))),
                'peak': float(np.max(np.abs(actual))),
                'fraction_abs_ge_1': float(np.mean(np.abs(actual) >= 1)),
                'decoded_seconds': len(actual) / rate})
        features[name] = np.stack(arrays); signal[name] = audits
        print(json.dumps({'condition': name, 'actual_mp3_clap_encoded': len(arrays)}), flush=True)
    tmp = target.with_suffix('.npz.tmp')
    with tmp.open('wb') as handle:
        np.savez(handle, text=text, **features)
    tmp.replace(target)
    evaluation.write_json(index, {'completed': True, 'context': context,
        'feature_sha256': generation.digest(target), 'clap_provenance': retrieve.model_provenance(module, args.device),
        'tokenizer_audit': token_audit, 'signal': signal})
    return read_clap(args.output_dir, context, ids, factors)


def read_clap(directory, context, ids, factors):
    target = directory / 'clap_features.npz'; meta = read_json(directory / 'clap_manifest.json')
    if (meta.get('completed') is not True or meta['context'] != context
            or meta['feature_sha256'] != generation.digest(target)
            or meta['clap_provenance']['checkpoint_sha256'] != retrieve.CLAP_SHA256):
        raise ValueError('CLAP cache differs from actual inputs, checkpoint, code or preprocessing')
    with np.load(target, allow_pickle=False) as archive:
        values = {k: archive[k].copy() for k in archive.files}
    expected = {'text'} | {c['id'] for c in factors}
    if set(values) != expected or len({v.shape for v in values.values()}) != 1:
        raise ValueError('CLAP cache has missing conditions or inconsistent dimensions')
    for value in values.values():
        if value.ndim != 2 or value.shape[0] != len(ids): raise ValueError('CLAP cache has incomplete cohort')
        retrieve.normalize_rows(value)
    for name in expected - {'text'}:
        if [r['id'] for r in meta['signal'][name]] != ids:
            raise ValueError('Signal audit is not aligned with the frozen cohort')
    return values, meta


def compare(args, protocol, ids, factors, records, settings, clap):
    import torch
    import torch.nn.functional as F
    _, _, seed, resamples = frozen_design(protocol)
    embeddings, clap_meta = clap
    retrieval = read_json(args.retrieval)
    if set(retrieval) != {str(i) for i in range(100)} or any(not isinstance(v, str) for v in retrieval.values()):
        raise ValueError('Retrieval must contain all 100 original caption IDs with string reference IDs')
    rv, rvi = evaluation.read_cache(args.reference_vgg, 'vggish')
    rp, rpi = evaluation.read_cache(args.reference_passt, 'passt')
    reference_hashes = {r['id']: r['source_sha256'] for r in rvi['records']}
    if any(reference_hashes.get(r['id']) != r['source_sha256'] for r in rpi['records']):
        raise ValueError('Reference VGGish and PaSST source audio differ')
    if any(retrieval[i] not in rv or retrieval[i] not in rp for i in ids):
        raise ValueError('Every cohort ID needs both encoders for its retrieved proxy')
    compute_fd, _ = evaluation.upstream()
    per_rows, summaries, metric_arrays = [], {}, {k: {} for k in METRICS}
    text = retrieve.normalize_rows(embeddings['text'])
    reference_logits = np.stack([rp[retrieval[i]] for i in ids])
    if reference_logits.shape != (len(ids), 527) or not np.isfinite(reference_logits).all():
        raise ValueError('Invalid reference PaSST shape or values')
    ref_log = F.log_softmax(torch.as_tensor(reference_logits, dtype=torch.float32), dim=1)
    for factor in factors:
        name = factor['id']; _, vg_dir, pa_dir = args.conditions[name]
        vg, vgi = evaluation.read_cache(vg_dir, 'vggish'); pa, pai = evaluation.read_cache(pa_dir, 'passt')
        if vgi['provenance'] != rvi['provenance'] or pai['provenance'] != rpi['provenance']:
            raise ValueError('Reference and generated encoders differ')
        for index in (vgi, pai):
            sources = {r['id']: r['source_sha256'] for r in index['records']}
            if any(sources.get(i) != records[name][i]['mp3']['sha256'] for i in ids):
                raise ValueError('Encoder cache did not use the exact cohort MP3 files')
        if any(i not in vg or i not in pa for i in ids):
            raise ValueError('A generated cohort encoder feature is missing')
        logits = np.stack([pa[i] for i in ids])
        if logits.shape != (len(ids), 527) or not np.isfinite(logits).all():
            raise ValueError('Invalid generated PaSST shape or values')
        for vector in [vg[i] for i in ids] + [rv[retrieval[i]] for i in ids]:
            if vector.ndim != 2 or vector.shape[1] != 128 or not len(vector) or not np.isfinite(vector).all():
                raise ValueError('Invalid VGGish frame vectors')
        kl = F.kl_div(F.log_softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1),
                      ref_log, reduction='none', log_target=True).sum(dim=1).numpy()
        cosine = np.sum(text * retrieve.normalize_rows(embeddings[name]), axis=1)
        signals = clap_meta['signal'][name]
        arrays = {'CLAP': cosine, 'KL-PASST-softmax-retrieved-proxy': kl,
                  **{k: np.array([r[k] for r in signals]) for k in ('rms', 'peak', 'fraction_abs_ge_1')}}
        if any(not np.isfinite(v).all() for v in arrays.values()): raise ValueError('Nonfinite cohort metric')
        fd = float(compute_fd(np.concatenate([vg[i] for i in ids]),
                              np.concatenate([rv[retrieval[i]] for i in ids])))
        if not np.isfinite(fd) or fd < -1e-6: raise ValueError('Subset FD failed; a sentinel is not a score')
        summaries[name] = {k: bootstrap_mean(v, seed, resamples) for k, v in arrays.items()}
        summaries[name]['FD-VGG-cohort-vs-retrieved-proxy'] = fd
        for k, v in arrays.items(): metric_arrays[k][name] = v
        for j, identifier in enumerate(ids):
            per_rows.append({'condition': name, 'id': identifier, 'reference_id': retrieval[identifier],
                'seed': records[name][identifier]['seed'], **{k: float(v[j]) for k, v in arrays.items()}})
    baseline = next(c['id'] for c in factors if not c['schedule'] and not c['rescale'])
    deltas, effects = {}, {}
    for metric, cells in metric_arrays.items():
        deltas[metric] = {name: bootstrap_mean(value - cells[baseline], seed, resamples)
                          for name, value in cells.items() if name != baseline}
        effects[metric] = {name: bootstrap_mean(value, seed, resamples)
                           for name, value in factorial_effects(cells, factors).items()}
        for row in per_rows:
            j = ids.index(row['id'])
            row[metric + '_delta_vs_baseline'] = float(cells[row['condition']][j] - cells[baseline][j])
    result = {'completed': True, 'analysis_type': 'predeclared paired 2x2 subset ablation',
        'not_the_required_full_100_clip_evaluation': True, 'cohort_ids': ids, 'n_captions': len(ids),
        'n_conditions': 4, 'reference_is_ground_truth': False,
        'reference_policy': 'Each caption uses its fixed CLAP top-1 retrieved proxy; KL(reference || generated).',
        'main_submission_variant_preselected': protocol['main_submission_variant'],
        'condition_settings': settings, 'summary': summaries, 'paired_deltas_vs_baseline': deltas,
        'factorial_effects': effects, 'per_caption': per_rows,
        'effect_definitions': {'schedule_main_effect': 'mean of schedule-on minus off across rescale levels',
            'rescale_main_effect': 'mean of rescale-on minus off across schedule levels',
            'interaction_difference_in_differences': 'combined - schedule_only - rescale_only + baseline'},
        'interpretation': {'CLAP': 'higher is closer global caption/audio alignment',
            'KL-PASST-softmax-retrieved-proxy': 'lower is closer to a retrieved proxy, not ground truth',
            'rms_peak_clipping': 'signal diagnostics, not musical quality or human preference',
            'confidence_intervals': 'Exploratory paired percentile bootstrap over captions; no multiple-comparison correction. One fixed seed per caption, so seed uncertainty is not estimated.',
            'subset_fd': 'Set-level descriptive score with a small cohort; no per-caption FD or artificial window sample count.'},
        'provenance': {'protocol_sha256': generation.digest(args.protocol),
            'captions_sha256': generation.digest(args.captions), 'retrieval_sha256': generation.digest(args.retrieval),
            'analysis_source_sha256': generation.digest(__file__),
            'clap_cache_manifest_sha256': generation.digest(args.output_dir / 'clap_manifest.json'),
            'reference_vgg_index_sha256': generation.digest(args.reference_vgg / 'index.json'),
            'reference_passt_index_sha256': generation.digest(args.reference_passt / 'index.json'),
            'generated_encoder_indices': {c['id']: {kind: generation.digest(path / 'index.json')
                for kind, path in zip(('vggish', 'passt'), args.conditions[c['id']][1:])} for c in factors}},
        'created_at_utc': datetime.now(timezone.utc).isoformat()}
    evaluation.write_json(args.output_dir / 'ablation_results.json', result)
    csv_path = args.output_dir / 'ablation_per_caption.csv'
    with csv_path.with_suffix('.csv.tmp').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_rows[0])); writer.writeheader(); writer.writerows(per_rows)
    csv_path.with_suffix('.csv.tmp').replace(csv_path)
    print(json.dumps({'completed': True, 'cohort_captions': len(ids), 'conditions': 4,
                      'output': str(args.output_dir / 'ablation_results.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('extract', 'compare', 'run'), default='run')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--captions', type=Path, default=ROOT / 'sample_code/test.csv')
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--condition', action='append', required=True, metavar='ID=GEN_DIR,VGG_CACHE,PASST_CACHE')
    parser.add_argument('--reference-vgg', type=Path, required=True)
    parser.add_argument('--reference-passt', type=Path, required=True)
    parser.add_argument('--clap-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        args.conditions = {}
        for value in args.condition:
            name, spec = value.split('=', 1); paths = spec.split(',')
            if name in args.conditions or len(paths) != 3 or any(not p for p in paths):
                raise ValueError('Each condition must occur once as ID=GEN_DIR,VGG_CACHE,PASST_CACHE')
            args.conditions[name] = tuple(Path(p) for p in paths)
        protocol, ids, factors, captions, _, records, settings = validate_group(
            args.protocol, args.captions, {k: v[0] for k, v in args.conditions.items()})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        context = context_for(args, ids, factors, records)
        clap = (extract_clap(args, context, ids, factors, captions, records)
                if args.mode in ('extract', 'run') else read_clap(args.output_dir, context, ids, factors))
        if args.mode in ('compare', 'run'): compare(args, protocol, ids, factors, records, settings, clap)
    except (ValueError, KeyError, OSError, RuntimeError, ImportError, TypeError) as exc:
        parser.exit(1, 'Ablation analysis stopped: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
