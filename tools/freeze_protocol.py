"""Freeze the HW3 decisions after real independent runtime probes, before tests.

This is intentionally one-shot. It does not run or inspect test-set scores.
The course baseline is preselected for submission; the other conditions form
a paired 2x2 study on the first 20 original IDs, selected without their scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import generate
import retrieve


def read(path):
    return json.loads(Path(path).read_text())


def freeze(args):
    if args.output.exists():
        raise ValueError('Preserve an existing protocol; do not rewrite decisions after test execution')
    pilot = read(args.pilot / 'manifest.json')
    identity = pilot['identity']
    samples = pilot['samples']
    if (pilot.get('complete') is not True or identity.get('mode') != 'development_pilot'
            or len(samples) != 1 or samples[0].get('official_waveform_parity', {}).get('passed') is not True):
        raise ValueError('A complete, independent, pretrained baseline waveform-parity pilot is required')
    sources = {name: generate.digest(ROOT / name) for name in ('generate.py', 'cfg.py')}
    if identity['source_sha256'] != sources:
        raise ValueError('Generation code changed after its real parity probe; run a new pilot')
    generate.verify_record(args.pilot, samples[0], samples[0]['contract_sha256'])
    settings = identity['settings']
    if (settings['guidance_scale'] != 7.0 or settings['schedule'] != 'constant'
            or settings['rescale'] != 0.0 or settings['steps'] != 100 or settings['seconds'] != 10.0):
        raise ValueError('This study fixes the course example baseline: CFG 7, 100 steps, 10 seconds')
    metadata = read(args.model / 'model_provenance.json')
    if metadata.get('completed') is not True or metadata['revision'] != settings['model_revision']:
        raise ValueError('The complete authorized model provenance must match the pilot')
    if generate.model_inventory(args.model) != identity['model_files']:
        raise ValueError('Model components changed after the pilot')
    smoke = read(args.clap_smoke / 'retrieval_manifest.json')
    if (smoke.get('status') != 'non_test_smoke_complete' or smoke.get('reference_count', 0) < 1
            or smoke.get('batch_size') != args.clap_batch_size
            or smoke['model']['checkpoint_sha256'] != retrieve.CLAP_SHA256
            or smoke['model']['retrieval_source_sha256'] != retrieve.sha256(ROOT / 'retrieve.py')):
        raise ValueError('Current CLAP code needs a real successful non-test smoke')
    encoders = read(args.encoder_smoke)
    if (encoders.get('status') != 'passed' or encoders.get('adapter_unchanged_during_probe') is not True
            or encoders.get('adapter_sha256_after') != generate.digest(ROOT / 'evaluate.py')
            or set(encoders.get('encoders', {})) != {'vggish', 'passt'}
            or not all(v.get('passed') is True for v in encoders['encoders'].values())):
        raise ValueError('Both real evaluation encoders must pass on the current adapter')
    rows = generate.read_captions(args.captions)
    _, inventory = retrieve.reference_inventory(args.references)
    variants = []
    for name, scheduled, rescaled in [
            ('baseline', False, False), ('schedule_only', True, False),
            ('rescale_only', False, True), ('combined', True, True)]:
        current = dict(settings)
        current.update(guidance_scale=10.0 if scheduled else 7.0,
                       schedule='linear' if scheduled else 'constant',
                       end_scale=4.0 if scheduled else 1.0, rescale=0.5 if rescaled else 0.0)
        generate.validate_settings(current)
        scales = [generate.guidance_at_step(current['guidance_scale'], i, current['steps'],
                  current['schedule'], current['end_scale']) for i in range(current['steps'])]
        if abs(sum(scales) / len(scales) - 7.0) > 1e-12:
            raise ValueError('The schedule must match the baseline arithmetic mean guidance')
        variants.append({'id': name, 'settings': current,
                         'planned_ids': [r['id'] for r in rows[:100 if name == 'baseline' else 20]],
                         'mean_guidance_over_steps': sum(scales) / len(scales)})
    development_probes = {}
    for variant in variants:
        if variant['id'] == 'baseline':
            continue
        path = args.pilot.parent / ('piano_' + variant['id'])
        probe = read(path / 'manifest.json')
        probe_identity = probe['identity']
        if (probe.get('complete') is not True or probe_identity.get('mode') != 'development_pilot'
                or probe_identity.get('settings') != variant['settings']
                or probe_identity.get('source_sha256') != sources
                or probe_identity.get('model_files') != identity['model_files']
                or probe_identity.get('runtime_versions') != identity['runtime_versions']
                or len(probe.get('samples', [])) != 1):
            raise ValueError('Each extra method needs an independent pretrained technical probe: ' + variant['id'])
        sample = probe['samples'][0]
        if (sample['caption_sha256'] != samples[0]['caption_sha256']
                or sample['seed'] != samples[0]['seed']
                or sample['sampling']['initial_latents_sha256_float32'] != samples[0]['sampling']['initial_latents_sha256_float32']):
            raise ValueError('Development probes must share the independently written prompt and initial noise')
        generate.verify_record(path, sample, sample['contract_sha256'])
        development_probes[variant['id']] = {
            'manifest_sha256': generate.digest(path / 'manifest.json'),
            'generation_seconds': sample['generation_seconds'], 'audio_audit': sample['audio_audit'],
            'purpose': 'Numerical/export/runtime check only; submission baseline was preselected, not a test-based winner.'}
    protocol = {
        'status': 'frozen', 'student_id': 'r14942154',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'no_test_or_validation_tuning': True,
        'test_csv_sha256': generate.digest(args.captions),
        'source_sha256': {**sources, 'retrieve.py': generate.digest(ROOT / 'retrieve.py'),
                          'evaluate.py': generate.digest(ROOT / 'evaluate.py')},
        'main_submission_variant': 'baseline',
        'selection_reason': 'Course baseline preselected before test results. No test-based winner selection.',
        'generation_conditions': variants,
        'seed_policy': 'base_seed + original integer caption ID; CPU RNG; shared across all conditions',
        'retrieval': {'checkpoint_sha256': retrieve.CLAP_SHA256,
                      'preprocess_policy_sha256': retrieve.object_hash(retrieve.PREPROCESS_POLICY),
                      'retrieval_source_sha256': retrieve.sha256(ROOT / 'retrieve.py'),
                      'captions_sha256': generate.digest(args.captions),
                      'reference_inventory_sha256': retrieve.object_hash(inventory),
                      'device': smoke['model']['device'], 'batch_size': args.clap_batch_size},
        'ablation': {
            'cohort_ids': [str(i) for i in range(20)],
            'selection': 'First 20 original CSV IDs, fixed before any test score; not a random population sample.',
            'conditions': [{'id': v['id'], 'schedule': v['settings']['schedule'] == 'linear',
                            'rescale': v['settings']['rescale'] > 0} for v in variants],
            'bootstrap': {'seed': 20261115, 'resamples': 10000},
            'hypotheses': [
                'Moving guidance across steps may change text alignment and signal artifacts at matched mean strength.',
                'Rescaling predicted v may alter alignment or artifacts; it is not waveform loudness normalization.',
                'The two changes may interact, so a combined condition alone cannot identify their effects.'],
            'limitations': [
                'Mean guidance over discrete steps does not match integration over sigma or conditional vector norms.',
                'One fixed noise realization per caption cannot estimate across-seed variability.',
                'Twenty IDs are a limited, nonrandom cohort. Paired intervals are descriptive, not human quality claims.',
                'The image-domain guidance literature does not establish improvements for this audio model.'],
            'references': ['https://arxiv.org/abs/2305.08891', 'https://arxiv.org/abs/2404.07724'],
            'method_relation': 'Prediction-std rescaling is adapted from prior work. The linear 10-to-4 audio schedule is an experimental heuristic, not the cited limited-interval algorithm.'},
        'evaluation': {
            'required': ['FD-VGG', 'KL-PASST-softmax', 'CLAP'],
            'score_actual_submitted_mp3': True, 'original_captions_unchanged': True,
            'fd_population': 'Report all 5367 references and 100 retrieved rows as distinct scores.',
            'kl_pairing': 'KL(reference || generated), fixed CLAP top-1 retrieval proxy by original caption ID.',
            'proxy_is_ground_truth': False, 'ta_reference_interpretation_confirmed': False,
            'coverage': 100,
            'clap_controls': ['Cyclic caption mismatch: generated ID i vs caption (i-1) mod 100.',
                              'One exact 10-second zero waveform vs all original captions.'],
            'mqascore': 'Optional; not claimed without a separately validated implementation.'},
        'probe_evidence': {'pilot_manifest_sha256': generate.digest(args.pilot / 'manifest.json'),
                           'extra_method_development_probes': development_probes,
                           'clap_smoke_manifest_sha256': generate.digest(args.clap_smoke / 'retrieval_manifest.json'),
                           'encoder_smoke_sha256': generate.digest(args.encoder_smoke),
                           'actual_pretrained_waveform_parity': samples[0]['official_waveform_parity'],
                           'pilot_generation_seconds_including_native_comparison': samples[0]['generation_seconds']},
        'deadline_utc': '2026-11-16T15:59:00+00:00',
        'compute_plan': '100 baseline clips plus 20 clips for each of three extra conditions = 160; baseline 100 has priority.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'frozen_protocol': str(args.output), 'sha256': generate.digest(args.output),
                      'main': 'baseline', 'planned_test_clips': 160}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--clap-smoke', type=Path, required=True)
    p.add_argument('--encoder-smoke', type=Path, default=ROOT / 'analysis/evaluator_encoder_smoke.json')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--captions', type=Path, default=ROOT / 'sample_code/test.csv')
    p.add_argument('--references', type=Path, default=ROOT / 'audio_mp3')
    p.add_argument('--clap-batch-size', type=int, default=4)
    p.add_argument('--output', type=Path, default=ROOT / 'analysis/experiment_protocol_frozen.json')
    a = p.parse_args()
    if a.clap_batch_size < 1:
        p.error('CLAP batch size must be positive')
    freeze(a)


if __name__ == '__main__':
    main()
