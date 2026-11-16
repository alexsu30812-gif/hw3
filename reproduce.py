"""Reproduce the frozen 100-clip baseline on another supported device.

Only device/dtype may change. Model bytes, captions, seeds, guidance, sampling
steps, duration and MP3 parameters come from the original verified experiment.
The runtime-derived protocol is explicitly labeled as a reproduction and keeps
the original method-selection timestamp. Cross-hardware bitwise equality is
not promised; the generator records the actual runtime and codec binaries.

Example, using the original protocol and completed baseline manifest:
  python reproduce.py --protocol analysis/experiment_protocol_frozen.json \
    --manifest results/baseline/manifest.json --captions /path/to/test.csv \
    --model-path /path/to/the/pinned/local/model --output-dir results/reproduced_cuda \
    --device cuda --dtype float16

Use the same command to resume this wrapper's own matching output directory.
Optionally provide both --reference-audio-dir and --clap-checkpoint to reproduce
the fixed CLAP retrieval too, in <output-dir>.reproduction/retrieval with a new
reference-embedding cache. Retrieval stays float32 with its frozen batch size;
very close scores can rank differently across devices.
No input or original submission is modified. No model is downloaded.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import generate
import retrieve

ROOT = Path(__file__).resolve().parent
RUNTIME_FIELDS = {'device', 'dtype'}
SETTING_FIELDS = {'model_repository', 'model_revision', 'guidance_scale', 'steps', 'seconds',
                  'base_seed', 'schedule', 'end_scale', 'rescale', 'device', 'dtype',
                  'generator_device', 'scheduler_rng_policy', 'mp3_bitrate'}
LABEL = 'runtime_reproduction_of_original_frozen_method'


def read_json(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_original(protocol_path, manifest_path, captions_path, model_path):
    """Verify the original complete manifest and the locally supplied model.

    Original generated audio is not required to rerun inference. This validates
    the full manifest's caption/seed contracts, not the existence of its MP3s.
    """
    protocol = read_json(protocol_path); manifest = read_json(manifest_path)
    if protocol.get('protocol_kind') == LABEL or 'original_protocol_sha256' in protocol:
        raise ValueError('Supply the original experiment protocol, not a derived reproduction protocol')
    if protocol.get('main_submission_variant') != 'baseline':
        raise ValueError('This wrapper reproduces the original preselected baseline')
    identity = manifest['identity']; settings = identity['settings']
    if set(settings) != SETTING_FIELDS:
        raise ValueError('Unrecognized generation settings require explicit wrapper review')
    generate.validate_settings(settings)
    if settings['model_repository'] != generate.MODEL_REPOSITORY or settings['generator_device'] != 'cpu':
        raise ValueError('Original model repository or fixed CPU-noise policy differs')
    if settings['schedule'] != 'constant' or settings['rescale'] != 0:
        raise ValueError('The original baseline must use constant CFG without experimental rescaling')
    checked = generate.validate_frozen_protocol(protocol_path, captions_path, settings)
    if checked['variant'] != 'baseline' or identity.get('protocol') != checked:
        raise ValueError('Manifest is not the baseline from the exact original frozen protocol')
    sources = {name: generate.digest(ROOT / name) for name in ('generate.py', 'cfg.py')}
    if identity.get('source_sha256') != sources:
        raise ValueError('Original generation/CFG source differs from the current source')
    if any(protocol.get('source_sha256', {}).get(name) != digest for name, digest in sources.items()):
        raise ValueError('The original protocol does not bind the same generation/CFG source')
    if (manifest.get('complete') is not True or manifest.get('formal_complete') is not True
            or manifest.get('completed_count') != 100 or identity.get('mode') != 'formal_test'
            or identity.get('csv_sha256') != generate.digest(captions_path)
            or manifest.get('run_sha256') != generate.text_sha(generate.canonical(identity))):
        raise ValueError('Require the original completed formal 100-caption baseline manifest')
    rows = generate.read_captions(captions_path)
    prompts = [{'id': r['id'], 'caption_sha256': generate.text_sha(r['caption'])} for r in rows]
    if identity.get('prompts') != prompts:
        raise ValueError('Original prompts do not match the exact original CSV')
    samples = {s['id']: s for s in manifest['samples']}
    if len(samples) != 100 or len(manifest['samples']) != 100 or set(samples) != {r['id'] for r in rows}:
        raise ValueError('Original manifest must contain each caption ID exactly once')
    for row in prompts:
        sample = samples[row['id']]; seed = settings['base_seed'] + int(row['id'])
        contract = generate.text_sha(generate.canonical({'run_sha256': manifest['run_sha256'],
                      'id': row['id'], 'caption_sha256': row['caption_sha256']}))
        if (sample.get('caption_sha256') != row['caption_sha256'] or sample.get('seed') != seed
                or sample.get('contract_sha256') != contract):
            raise ValueError('An original caption/seed contract changed: ' + row['id'])
    model_files = generate.model_inventory(model_path)
    if identity.get('model_files') != model_files:
        raise ValueError('Local model/config bytes differ from the original model inventory')
    return protocol, manifest, settings, model_files


def derive_protocol(original, device, dtype, original_sha, manifest_sha, recorded_at, *, reproduce_retrieval=False):
    if device not in {'cpu', 'mps', 'cuda'} or dtype not in {'float16', 'float32'}:
        raise ValueError('Use an explicit supported device and dtype')
    if device == 'cpu' and dtype == 'float16':
        raise ValueError('CPU reproduction requires float32')
    derived = copy.deepcopy(original)
    for condition, original_condition in zip(derived['generation_conditions'], original['generation_conditions']):
        before = original_condition['settings']
        if set(before) != SETTING_FIELDS:
            raise ValueError('Unexpected frozen setting keys require wrapper review')
        condition['settings'].update(device=device, dtype=dtype)
        if {k: v for k, v in before.items() if k not in RUNTIME_FIELDS} != {
                k: v for k, v in condition['settings'].items() if k not in RUNTIME_FIELDS}:
            raise ValueError('Reproduction may only replace device and dtype')
    if reproduce_retrieval:
        before = original['retrieval']
        derived['retrieval']['device'] = device
        if {k: v for k, v in before.items() if k != 'device'} != {
                k: v for k, v in derived['retrieval'].items() if k != 'device'}:
            raise ValueError('Retrieval reproduction may only replace device')
    derived.update(protocol_kind=LABEL, original_protocol_sha256=original_sha,
                   original_baseline_manifest_sha256=manifest_sha,
                   runtime_override_recorded_at_utc=recorded_at,
                   reproduction={
                       'purpose': 'Only move device/dtype; do not select or change the method from test results.',
                       'original_method_selection_frozen_at_utc': original['frozen_at_utc'],
                       'this_is_not_a_new_original_freeze_event': True,
                       'only_generation_settings_overrides': ['device', 'dtype'],
                       'target_device': device, 'target_dtype': dtype,
                       'retrieval_reproduction_requested': reproduce_retrieval,
                       'only_retrieval_settings_overrides': ['device'] if reproduce_retrieval else [],
                       'retrieval_precision': 'float32, fixed by the original retrieval implementation',
                       'retrieval_numerical_limitations': 'Model embeddings can differ slightly across devices. Very close cosine scores may change ranking; exact-score lexical tie rules are unchanged.',
                       'cross_hardware_bitwise_equality_guaranteed': False,
                       'codec_policy': 'Preserve original MP3 arguments/bitrate. Record the actual platform codec and runtime in the new manifest.'})
    return derived


def retrieval_requested(reference_audio_dir, clap_checkpoint):
    if (reference_audio_dir is None) != (clap_checkpoint is None):
        raise ValueError('--reference-audio-dir and --clap-checkpoint must be provided together')
    return reference_audio_dir is not None


def validate_retrieval_inputs(protocol_path, captions_path, reference_audio_dir, clap_checkpoint):
    """Check the exact original reference population and required CLAP bytes."""
    original = read_json(protocol_path)
    frozen = original['retrieval']
    batch = frozen.get('batch_size')
    if type(batch) is not int or batch < 1:
        raise ValueError('The original positive retrieval batch size is required')
    _, inventory = retrieve.reference_inventory(reference_audio_dir, expected_count=5367)
    if len(inventory) != 5367:
        raise ValueError('Retrieval reproduction requires all original 5367 references')
    retrieve.check_frozen_protocol(protocol_path, captions_path, inventory,
                                   device=frozen.get('device'), batch_size=batch)
    if (clap_checkpoint.name != retrieve.CLAP_FILE
            or clap_checkpoint.stat().st_size != retrieve.CLAP_BYTES
            or retrieve.sha256(clap_checkpoint) != retrieve.CLAP_SHA256):
        raise ValueError('The local CLAP checkpoint must match the exact course music model')
    return inventory, {'reference_inventory_sha256': retrieve.object_hash(inventory),
        'reference_count': len(inventory), 'checkpoint_sha256': retrieve.CLAP_SHA256,
        'preprocess_policy_sha256': retrieve.object_hash(retrieve.PREPROCESS_POLICY),
        'retrieval_source_sha256': retrieve.sha256(ROOT / 'retrieve.py'),
        'batch_size': batch, 'precision': 'float32'}


def retrieval_command(python, captions, reference_audio, checkpoint, output, cache, protocol, frozen, device):
    return [str(python), str(ROOT / 'retrieve.py'), '--checkpoint', str(checkpoint),
            '--audio-dir', str(reference_audio), '--captions', str(captions),
            '--output-dir', str(output), '--cache-dir', str(cache), '--protocol', str(protocol),
            '--device', device, '--batch-size', str(frozen['batch_size']),
            '--expected-reference-count', '5367', '--mode', 'retrieve']


def validate_retrieval_result(output, protocol_path, captions_path, inventory, frozen, device):
    manifest_path = output / 'retrieval_manifest.json'; prediction_path = output / 'retrieval.json'
    manifest = read_json(manifest_path); prediction = read_json(prediction_path)
    retrieve.validate_prediction(prediction, [r['id'] for r in inventory])
    model = manifest['model']
    if (manifest.get('status') != 'retrieval_complete' or manifest.get('mode') != 'retrieve'
            or manifest.get('reference_count') != 5367 or manifest.get('caption_count') != 100
            or manifest.get('batch_size') != frozen['batch_size']
            or manifest.get('caption_source_sha256') != generate.digest(captions_path)
            or manifest.get('protocol_sha256') != generate.digest(protocol_path)
            or manifest.get('retrieval_json_sha256') != generate.digest(prediction_path)
            or manifest.get('preprocessing') != retrieve.PREPROCESS_POLICY
            or model.get('checkpoint_sha256') != retrieve.CLAP_SHA256
            or model.get('device') != device or model.get('precision') != 'float32'
            or model.get('retrieval_source_sha256') != retrieve.sha256(ROOT / 'retrieve.py')
            or [r['source'] for r in manifest.get('records', [])] != inventory):
        raise ValueError('Reproduced retrieval does not match the fixed method, inputs or complete coverage')
    return {'manifest_sha256': generate.digest(manifest_path), 'retrieval_json_sha256': generate.digest(prediction_path),
            'reference_count': 5367, 'caption_count': 100}


def generator_command(python, captions, model, output, protocol, settings):
    """Every algorithm/codec option is taken from the verified frozen settings."""
    return [str(python), str(ROOT / 'generate.py'), '--captions', str(captions),
            '--model-path', str(model), '--revision', settings['model_revision'],
            '--output-dir', str(output), '--protocol', str(protocol),
            '--device', settings['device'], '--dtype', settings['dtype'],
            '--scale', str(settings['guidance_scale']), '--steps', str(settings['steps']),
            '--seconds', str(settings['seconds']), '--base-seed', str(settings['base_seed']),
            '--schedule', settings['schedule'], '--end-scale', str(settings['end_scale']),
            '--rescale', str(settings['rescale']), '--mp3-bitrate', settings['mp3_bitrate']]


def protect_paths(output, control, protocol, manifest, captions, model):
    for directory in (output, control):
        if directory.is_symlink():
            raise ValueError('Reproduction output/control directories must not be symlinks')
        if (directory == model or model in directory.parents
                or directory == manifest.parent or manifest.parent in directory.parents):
            raise ValueError('Choose output/control outside the original generation and model directories')
        if any(directory == p or directory in p.parents for p in (protocol, manifest, captions, model)):
            raise ValueError('Reproduction must not contain or overwrite any original input')
    if output == control or output in control.parents or control in output.parents:
        raise ValueError('Output and reproduction control directories must be separate siblings')


def run(args):
    with_retrieval = retrieval_requested(args.reference_audio_dir, args.clap_checkpoint)
    for name in ('protocol', 'manifest', 'captions', 'model_path', 'output_dir'):
        path = getattr(args, name)
        if name == 'output_dir' and path.is_symlink():
            raise ValueError('Output directory must not be a symlink')
        setattr(args, name, path.resolve())
    output = args.output_dir
    control = output.with_name(output.name + '.reproduction')
    protect_paths(output, control, args.protocol, args.manifest, args.captions, args.model_path)
    if with_retrieval:
        args.reference_audio_dir = args.reference_audio_dir.resolve()
        args.clap_checkpoint = args.clap_checkpoint.resolve()
        for directory in (output, control):
            if (directory == args.reference_audio_dir or args.reference_audio_dir in directory.parents
                    or directory in args.reference_audio_dir.parents
                    or directory == args.clap_checkpoint or directory in args.clap_checkpoint.parents):
                raise ValueError('Choose reproduction paths outside the original retrieval inputs')
    original, original_manifest, original_settings, inventory = validate_original(
        args.protocol, args.manifest, args.captions, args.model_path)
    device, dtype, _ = generate.resolve_device(args.device, args.dtype)
    runtime_settings = dict(original_settings, device=device, dtype=dtype)
    reference_inventory, retrieval_identity = None, None
    if with_retrieval:
        reference_inventory, retrieval_identity = validate_retrieval_inputs(
            args.protocol, args.captions, args.reference_audio_dir, args.clap_checkpoint)
        retrieval_identity['device'] = device
    identity = {'schema': 'hw3_portable_baseline_reproduction_v2',
        'original_protocol_sha256': generate.digest(args.protocol),
        'original_baseline_manifest_sha256': generate.digest(args.manifest),
        'captions_sha256': generate.digest(args.captions),
        'model_inventory_sha256': generate.text_sha(generate.canonical(inventory)),
        'wrapper_sha256': generate.digest(__file__), 'runtime_settings': runtime_settings,
        'output_directory': str(output), 'retrieval': retrieval_identity}
    if output.exists() and any(output.iterdir()) and not (control / 'run_record.json').is_file():
        raise ValueError('Existing output is not owned by this reproduction wrapper; preserve it')
    if control.exists() and any(control.iterdir()) and not (control / 'run_record.json').is_file():
        raise ValueError('Existing control directory is unrecognized; preserve it')
    control.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (control / '.reproduction.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another wrapper is using this reproduction directory') from exc
        record_path = control / 'run_record.json'; derived_path = control / 'runtime_protocol.json'
        if record_path.exists():
            record = read_json(record_path)
            if record.get('identity') != identity:
                raise ValueError('Existing reproduction belongs to other inputs, source or runtime settings')
            derived = read_json(derived_path)
            expected = derive_protocol(original, device, dtype, identity['original_protocol_sha256'],
                identity['original_baseline_manifest_sha256'], record['runtime_override_recorded_at_utc'],
                reproduce_retrieval=with_retrieval)
            if derived != expected or generate.digest(derived_path) != record['derived_protocol_sha256']:
                raise ValueError('The runtime-derived protocol changed after this reproduction began')
        else:
            recorded_at = now()
            derived = derive_protocol(original, device, dtype, identity['original_protocol_sha256'],
                                      identity['original_baseline_manifest_sha256'], recorded_at,
                                      reproduce_retrieval=with_retrieval)
            with derived_path.open('x') as handle:
                json.dump(derived, handle, indent=2, ensure_ascii=False, allow_nan=False); handle.write('\n')
            record = {'identity': identity, 'protocol_kind': LABEL,
                'runtime_override_recorded_at_utc': recorded_at,
                'derived_protocol_sha256': generate.digest(derived_path),
                'original_audio_required': False,
                'validation_scope': 'Original full manifest contracts and local model bytes; no claim that original audio was downloaded or inspected.',
                'cross_hardware_bitwise_equality_guaranteed': False,
                'reproduction_complete': False, 'generation_complete': False,
                'retrieval_complete': False if with_retrieval else None, 'attempts': []}
            generate.write_json(record_path, record)
        if output.exists() and (output / 'manifest.json').exists():
            existing = read_json(output / 'manifest.json')
            expected = generate.validate_frozen_protocol(derived_path, args.captions, runtime_settings)
            if (existing['identity'].get('protocol') != expected
                    or existing['identity'].get('settings') != runtime_settings):
                raise ValueError('Existing output is not this wrapper\'s matching runtime reproduction')
        command = generator_command(sys.executable, args.captions, args.model_path, output, derived_path, runtime_settings)
        record['reproduction_complete'] = False
        record['attempts'].append({'started_at_utc': now(), 'generation_command': command, 'exit_code': None})
        generate.write_json(record_path, record)
        print(json.dumps({'protocol_kind': LABEL, 'runtime_protocol': str(derived_path),
                          'device': device, 'dtype': dtype,
                          'retrieval_requested': with_retrieval,
                          'cross_hardware_bitwise_equality_guaranteed': False}), flush=True)
        if with_retrieval:
            retrieval_output = control / 'retrieval'; retrieval_cache = control / 'clap_reference_cache'
            if retrieval_output.is_symlink() or retrieval_cache.is_symlink():
                raise ValueError('Reproduction retrieval paths must not be symlinks')
            retrieval_cmd = retrieval_command(sys.executable, args.captions, args.reference_audio_dir,
                args.clap_checkpoint, retrieval_output, retrieval_cache, derived_path, original['retrieval'], device)
            record['attempts'][-1]['retrieval_command'] = retrieval_cmd
            record['retrieval_complete'] = False
            generate.write_json(record_path, record)
            retrieval_run = subprocess.run(retrieval_cmd, check=False)
            record['attempts'][-1]['retrieval_exit_code'] = retrieval_run.returncode
            if retrieval_run.returncode != 0:
                record['attempts'][-1].update(finished_at_utc=now(), exit_code=retrieval_run.returncode)
                generate.write_json(record_path, record)
                return retrieval_run.returncode
            record['retrieval_result'] = validate_retrieval_result(retrieval_output, derived_path,
                args.captions, reference_inventory, original['retrieval'], device)
            record['retrieval_complete'] = True
            generate.write_json(record_path, record)
        completed = subprocess.run(command, check=False)
        record['attempts'][-1].update(finished_at_utc=now(), exit_code=completed.returncode)
        if completed.returncode == 0:
            result = read_json(output / 'manifest.json')
            record['generation_complete'] = bool(result.get('formal_complete') is True
                and result.get('completed_count') == 100 and result['identity']['settings'] == runtime_settings
                and result['identity']['protocol']['sha256'] == generate.digest(derived_path)
                and all(result['identity'].get(k) == original_manifest['identity'].get(k)
                        for k in ('source_sha256', 'model_files', 'csv_sha256', 'prompts')))
            if not record['generation_complete']:
                record['attempts'][-1]['validation_error'] = 'Subprocess did not produce a complete matching 100-clip manifest'
        else:
            record['generation_complete'] = False
        record['reproduction_complete'] = record['generation_complete'] and (
            not with_retrieval or record['retrieval_complete'] is True)
        generate.write_json(record_path, record)
        if completed.returncode != 0: return completed.returncode
        if not record['reproduction_complete']: raise ValueError('Reproduction output failed the final coverage check')
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True, help='Original completed 100-clip baseline manifest')
    parser.add_argument('--captions', type=Path, required=True, help='User-supplied original course test.csv')
    parser.add_argument('--model-path', type=Path, required=True, help='Complete local model matching original hashes')
    parser.add_argument('--output-dir', type=Path, required=True, help='New output, or this wrapper\'s own matching resume')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), required=True)
    parser.add_argument('--dtype', choices=('float16', 'float32'), required=True)
    parser.add_argument('--reference-audio-dir', type=Path,
                        help='Optional: original 5367 MusicCaps MP3s; requires --clap-checkpoint')
    parser.add_argument('--clap-checkpoint', type=Path,
                        help='Optional: exact required CLAP music checkpoint; requires --reference-audio-dir')
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, 'Reproduction stopped: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
