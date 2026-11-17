"""Build a small local source/results bundle; never publish or include datasets.

Final mode requires the genuine 100-clip submission, mandatory metrics, and a
validated final PDF. The default full mode additionally requires the predeclared
4x20 ablation. --baseline-only excludes all extra experiment artifacts, without
relaxing baseline checks. --draft only makes an explicitly incomplete preview.
This utility uses only the standard library and never reads authentication files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDENT = 'r14942154'
CONDITIONS = ('baseline', 'schedule_only', 'rescale_only', 'combined')
CLAP_SHA = 'fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd'
SAO_REVISION = 'f21265c1e2710b3bd2386596943f0007f55f802e'
SOURCE_FILES = ('cfg.py', 'generate.py', 'retrieve.py', 'evaluate.py', 'reproduce.py',
                'package_submission.py', 'README.md', 'requirements.txt',
                'requirements-lock.txt', 'requirements-report.txt', 'requirements-report-lock.txt',
                'analysis/compare_ablation.py')
TOOL_FILES = ('check_model_access.py', 'download_clap.py', 'download_ranges.py',
              'prepare_evaluation_models.py',
              'prepare_sao.py', 'freeze_protocol.py', 'build_report.py', 'build_report_analysis.py',
              'probe_runtime.py', 'check_vggish_mps.py',
              'smoke_evaluator_encoders.py', 'package_source.py',
              'finalize_when_ready.py', 'sync_progress.py', 'hw3_progress.py')
DENIED = {'.git', '.venv', '__pycache__', 'cache', 'caches', 'data', 'weights',
          'node_modules', '.aws', '.codex', '.agents', 'stored_tokens'}
TEXT_SUFFIXES = {'.py', '.md', '.txt', '.json', '.toml', '.cfg', '.yml', '.yaml'}
MAX_FILE = 12 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe_source(path, *, max_bytes=MAX_FILE):
    """Reject symlinks in every ancestor before resolving an allowed source."""
    path = Path(path).absolute()
    try:
        relative = path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError('A selected source is outside the HW3 directory') from exc
    current = ROOT
    if current.is_symlink():
        raise ValueError('The project root must not be a symlink')
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError('Symlink source rejected: ' + str(relative))
    if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Missing or escaping source: ' + str(relative))
    if path.stat().st_size > max_bytes:
        raise ValueError('Unexpectedly large selected file: ' + str(relative))
    return path


def safe_member(name):
    path = Path(name)
    if (not name or path.is_absolute() or '..' in path.parts
            or any(p.startswith('.') or p in DENIED for p in path.parts)
            or path.suffix.lower() in {'.pth', '.pt', '.bin', '.safetensors', '.wav', '.csv'}
            and name != 'results/ablation/ablation_per_caption.csv'):
        raise ValueError('Non-whitelisted archive path: ' + name)
    return path


def check(condition, message):
    if not condition:
        raise ValueError(message)


def validate_artifacts(args, directories):
    """Validate actual artifact bytes, not filenames or synthetic score fixtures."""
    protocol = read(safe_source(args.protocol))
    check(protocol.get('status') == 'frozen' and protocol.get('no_test_or_validation_tuning') is True,
          'A genuine frozen no-test-tuning protocol is required')
    check(protocol.get('main_submission_variant') == 'baseline', 'This bundle expects the preselected baseline')
    check(protocol.get('source_sha256') == {name: sha(ROOT / name) for name in
          ('generate.py', 'cfg.py', 'retrieve.py', 'evaluate.py')}, 'Code differs from the frozen protocol')
    settings = {c['id']: c['settings'] for c in protocol['generation_conditions']}
    cohort = protocol.get('ablation', {}).get('cohort_ids')
    check(set(settings) == set(CONDITIONS) and cohort == [str(i) for i in range(20)],
          'Require the predeclared four conditions on original IDs 0..19')
    expected = {str(i) for i in range(100)}
    conditions = ('baseline',) if args.baseline_only else CONDITIONS
    records, identities, noise = {}, {}, {}
    for name in conditions:
        directory = directories[name]
        manifest = read(safe_source(directory / 'manifest.json'))
        identity = manifest['identity']
        rows = manifest['samples']; keyed = {r['id']: r for r in rows}
        wanted = expected if name == 'baseline' else set(cohort)
        check(len(rows) == len(wanted) and set(keyed) == wanted, 'Missing or duplicate genuine clips: ' + name)
        check(identity.get('mode') == 'formal_test' and identity.get('settings') == settings[name]
              and identity.get('csv_sha256') == protocol['test_csv_sha256']
              and identity.get('protocol', {}).get('sha256') == sha(args.protocol)
              and identity['protocol'].get('variant') == name,
              'Generation does not match the frozen study: ' + name)
        check(manifest['run_sha256'] == text_sha(canonical(identity)), 'Generation identity hash changed: ' + name)
        check(identity.get('source_sha256') == {n: sha(ROOT / n) for n in ('generate.py', 'cfg.py')},
              'Generated clips used different source code: ' + name)
        if name == 'baseline':
            check(manifest.get('formal_complete') is True and manifest.get('complete') is True,
                  'The main 100-clip run is not complete')
        for identifier, row in keyed.items():
            contract = text_sha(canonical({'run_sha256': manifest['run_sha256'], 'id': identifier,
                                           'caption_sha256': row['caption_sha256']}))
            check(row.get('contract_sha256') == contract and row.get('seed') == settings[name]['base_seed'] + int(identifier),
                  'Caption/seed contract failed: ' + name + '/' + identifier)
            mp3 = row['mp3']; path = safe_source(directory / mp3['path'])
            check(mp3['path'] == f'generated_samples/{identifier}.mp3' and sha(path) == mp3['sha256'],
                  'Recorded MP3 bytes changed: ' + name + '/' + identifier)
            audit = row.get('audio_audit', {})
            check(audit.get('codec') == 'mp3' and audit.get('sample_rate') == 44100
                  and audit.get('channels') == 2 and audit.get('decoded_frames', 0) > 0,
                  'Missing actual MP3 decode evidence: ' + name + '/' + identifier)
            if identifier in cohort:
                value = {'seed': row['seed'], 'caption': row['caption_sha256'],
                         'noise': row['sampling']['initial_latents_sha256_float32'],
                         'scheduler': row['sampling']['scheduler']}
                check(identifier not in noise or noise[identifier] == value, 'Conditions do not share caption/noise/scheduler')
                noise[identifier] = value
        records[name], identities[name] = keyed, identity
    invariant = ('model_files', 'runtime_versions', 'source_sha256', 'codecs', 'prompts')
    check(all({k: identities[n][k] for k in invariant} == {k: identities['baseline'][k] for k in invariant}
              for n in conditions), 'A model/runtime/input control changed across conditions')
    retrieval = read(safe_source(args.retrieval_dir / 'retrieval.json'))
    meta = read(safe_source(args.retrieval_dir / 'retrieval_manifest.json'))
    check(set(retrieval) == expected and all(isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_-]{11}', v)
                                            for v in retrieval.values()), 'Invalid retrieval JSON IDs')
    check(meta.get('status') == 'retrieval_complete' and meta.get('reference_count') == 5367
          and meta.get('caption_count') == 100 and meta['model']['checkpoint_sha256'] == CLAP_SHA
          and meta.get('protocol_sha256') == sha(args.protocol)
          and meta.get('retrieval_json_sha256') == sha(args.retrieval_dir / 'retrieval.json'),
          'Retrieval is incomplete or inconsistent')
    metric = read(safe_source(args.evaluation))
    check(metric.get('completed') is True and metric['counts']['generated_clips'] == 100
          and metric['counts']['fd_reference_clips'] == 5367 and metric['counts']['paired_reference_rows'] == 100
          and metric.get('retrieval_sha256') == sha(args.retrieval_dir / 'retrieval.json')
          and metric.get('caption_csv_sha256') == protocol['test_csv_sha256'], 'Required full evaluation is incomplete')
    for name in ('FD-VGG-all-reference', 'FD-VGG-retrieved-proxy', 'KL-PASST-softmax-retrieved-proxy', 'CLAP'):
        value = metric.get(name)
        check(type(value) in (float, int) and math.isfinite(value) and (name == 'CLAP' or value >= 0),
              'Missing or invalid required metric: ' + name)
    check({r['id']: r['source_sha256'] for r in metric['clap_audio_inputs']} ==
          {i: r['mp3']['sha256'] for i, r in records['baseline'].items()}, 'Metrics evaluated different MP3 bytes')
    check(metric.get('clap_provenance', {}).get('checkpoint_sha256') == CLAP_SHA,
          'Required evaluation used a different CLAP checkpoint')
    if not args.baseline_only:
        ablation = read(safe_source(args.ablation_dir / 'ablation_results.json'))
        keyed = {(r['condition'], r['id']): r for r in ablation.get('per_caption', [])}
        check(ablation.get('completed') is True and ablation.get('n_conditions') == 4
              and ablation.get('n_captions') == 20 and ablation.get('cohort_ids') == cohort
              and len(keyed) == len(ablation['per_caption']) == 80
              and set(keyed) == {(n, i) for n in CONDITIONS for i in cohort}
              and ablation.get('condition_settings') == settings
              and ablation['provenance']['protocol_sha256'] == sha(args.protocol)
              and ablation['provenance']['captions_sha256'] == protocol['test_csv_sha256']
              and ablation['provenance']['retrieval_sha256'] == sha(args.retrieval_dir / 'retrieval.json')
              and ablation['provenance']['analysis_source_sha256'] == sha(ROOT / 'analysis/compare_ablation.py'),
              'The four-by-twenty ablation is incomplete or used changed inputs/code')
        with safe_source(args.ablation_dir / 'ablation_per_caption.csv').open(newline='') as handle:
            csv_rows = list(csv.DictReader(handle))
        check(len(csv_rows) == 80 and {(r['condition'], r['id']) for r in csv_rows} == set(keyed), 'Ablation CSV coverage mismatch')
        for row in csv_rows:
            source = keyed[row['condition'], row['id']]
            for key in ('CLAP', 'KL-PASST-softmax-retrieved-proxy', 'rms', 'peak', 'fraction_abs_ge_1'):
                check(type(source.get(key)) in (int, float) and math.isfinite(source[key])
                      and math.isfinite(float(row[key])) and abs(float(row[key]) - source[key]) < 1e-8,
                      'Ablation JSON/CSV numerical mismatch: ' + key)
    report = safe_source(args.report)
    checks = read(safe_source(report.with_suffix('.checks.json')))
    check(report.name == STUDENT + '_report.pdf' and report.read_bytes().startswith(b'%PDF-')
          and checks.get('sha256') == sha(report) and checks.get('draft') is False
          and checks.get('formal_result_validation_passed') is True and not checks.get('remaining_issues')
          and checks.get('aspect') == '16:9', 'A validated final report is required')
    scope = 'baseline_only' if args.baseline_only else 'full_with_ablation'
    check(checks.get('report_scope') == scope, 'Report validation scope differs from the requested source bundle')
    for entry in checks.get('inputs', {}).values():
        path = safe_source(Path(entry['path']))
        check(sha(path) == entry['sha256'], 'An input changed after the final report was validated')
    builder = checks.get('builder_path', 'tools/build_report.py')
    check(builder in {'tools/build_report.py', 'tools/build_report_analysis.py'}, 'Unexpected report builder path')
    check(checks.get('builder_sha256') == sha(safe_source(ROOT / builder)), 'Report builder changed after PDF validation')
    # The independently verified COOL ZIP is around 24 MiB and is not copied
    # into this source archive. Keep the normal 12 MiB limit for selected files.
    cool = safe_source(args.cool_zip, max_bytes=MAX_TOTAL - 1)
    members = {f'{STUDENT}/cfg.py': sha(ROOT / 'cfg.py'),
               f'{STUDENT}/retrieval.json': sha(args.retrieval_dir / 'retrieval.json')}
    members.update({f'{STUDENT}/generated_samples/{i}.mp3': r['mp3']['sha256']
                    for i, r in records['baseline'].items()})
    verify_zip(cool, members)
    model = read(safe_source(args.model_provenance))
    check(model.get('completed') is True and model.get('revision') == SAO_REVISION
          and any(r['path'].startswith('projection_model/') for r in model.get('files', [])),
          'Complete authorized model provenance including projection_model is required')
    inventory = {r['path']: {'bytes': r['bytes'], 'sha256': r['actual_sha256']} for r in model['files']
                 if Path(r['path']).suffix in {'.json', '.txt', '.model', '.safetensors', '.bin', '.spm'}}
    check(inventory == identities['baseline']['model_files'], 'Published model provenance differs from actual inference weights/configs')
    return {'validated': True, 'scope': scope, 'main_clips': 100,
            'extra_clips': 0 if args.baseline_only else 60,
            'ablation_rows': 0 if args.baseline_only else 80,
            'cool_zip_sha256': sha(cool), 'report_sha256': sha(report)}


def verify_zip(path, expected):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        check(len(names) == len(set(names)) and set(names) == set(expected), 'ZIP membership differs from the exact whitelist')
        check(archive.testzip() is None, 'ZIP CRC readback failed')
        for name in names:
            p = Path(name)
            check(not p.is_absolute() and '..' not in p.parts, 'Unsafe ZIP member')
            check((archive.getinfo(name).external_attr >> 16) & 0o170000 != 0o120000, 'ZIP symlink rejected')
            check(hashlib.sha256(archive.read(name)).hexdigest() == expected[name], 'ZIP SHA readback failed: ' + name)


def public_readme(original, draft, baseline_only=False):
    prefix = ('# DRAFT — incomplete local review bundle\n\nNo final submission is ready.\n\n' if draft else '')
    prefix += '''# Portable inference and artifact layout

This folder contains source and recorded results, not the released dataset or
model caches. Supply the course caption CSV and MusicCaps audio yourself. No
model was trained for this assignment; the gated pretrained Stable Audio Open
weights must be obtained from their official pinned revision after personally
accepting its terms. SAO/CLAP downloads verify pinned HF LFS full hashes;
evaluation downloads verify pinned mirror SHA256 and official release size,
because an independent full digest of the old official release is unavailable. Public
model provenance is under `model_provenance/`; no authentication token is bundled.

Install `requirements.txt` in a Python 3.11 virtual environment. For the report,
use a separate environment with `requirements-report.txt` and a local Traditional
Chinese font. Download models with `tools/prepare_sao.py --download` and
`tools/download_clap.py --download` after local HF login where needed.

Pass `--captions /path/to/your/course_captions.csv` explicitly to generation,
retrieval, analysis and report commands; pass `--audio-dir /path/to/musiccaps`
to retrieval/extraction. The supplied CSV must match the SHA recorded in the
frozen protocol. The CSV is intentionally absent from this public folder.
Generation also needs `--protocol analysis/experiment_protocol_frozen.json`
and the exact condition settings recorded there. Set a NEW output directory,
such as `reproduced/baseline`; the compact recorded results omit WAV masters
and are evidence, not resumable working caches. Use the original device/runtime
for the closest reproducibility; same seeds do not promise cross-device byte
equality. Do not overwrite the frozen study to select a method using test scores.

The official COOL ZIP is a separate deliverable. This source archive is for the
public project folder and is not the file uploaded to the COOL ZIP slot.

---

'''
    scope = ('This bundle has **baseline_only** scope: the 100-clip baseline and\n'
             'required evaluation are its only result artifacts. Extra experiment\n'
             'methods remain in the source/protocol for reproducibility, but no\n'
             'completed extra results or ablation are claimed or included.\n\n'
             if baseline_only else
             'This bundle has **full_with_ablation** scope, including the\n'
             'predeclared four-condition paired 20-caption comparison.\n\n')
    return prefix + scope + original.replace('sample_code/test.csv', '/path/to/your/course_captions.csv')


def package(args):
    check(not args.baseline_only or not args.condition, '--baseline-only cannot include extra --condition paths')
    conditions = ('baseline',) if args.baseline_only else CONDITIONS
    scope = 'baseline_only' if args.baseline_only else 'full_with_ablation'
    directories = {name: ROOT / 'results' / name for name in conditions}
    directories['baseline'] = args.generation_dir
    for raw in args.condition:
        name, separator, value = raw.partition('=')
        check(separator and name in CONDITIONS and name != 'baseline', 'Use --condition NAME=PATH for an extra condition')
        directories[name] = Path(value).absolute()
    validation = {'validated': False, 'reason': 'DRAFT: formal artifacts were not validated'}
    if not args.draft:
        validation = validate_artifacts(args, directories)
    output = args.output_dir or ROOT / 'deliverables' / (STUDENT + '_source'
             + ('_BASELINE' if args.baseline_only else '') + ('_DRAFT' if args.draft else ''))
    output = output.absolute()
    check(output.is_relative_to(ROOT) and not output.is_symlink(), 'Output must be a nonsymlink directory inside HW3')
    current = ROOT
    for part in output.relative_to(ROOT).parts:
        current /= part
        check(part not in {'.', '..'} and not current.is_symlink(), 'Output ancestors cannot escape or be symlinks')
    check(output.resolve().is_relative_to(ROOT.resolve()), 'Resolved output escapes HW3')
    check(args.draft == ('draft' in output.name.lower()), 'DRAFT must appear in draft output names and be absent in final names')
    check(not output.exists() and not output.with_suffix('.zip').exists(), 'Preserve existing review bundles; choose a new output directory')
    selected, missing = {}, []

    def add(source, destination, required=True):
        destination = str(safe_member(destination))
        if not Path(source).is_file():
            if required and not args.draft:
                raise ValueError('Missing required bundle artifact: ' + destination)
            if required:
                missing.append(destination)
            return
        check(destination not in selected, 'Duplicate selected bundle destination')
        selected[destination] = safe_source(source)

    for name in SOURCE_FILES:
        if name != 'README.md':
            add(ROOT / name, name)
    for name in TOOL_FILES:
        add(ROOT / 'tools' / name, 'tools/' + name)
    for vendor in ('av-benchmark', 'passt_hear21'):
        base = ROOT / 'vendor' / vendor
        check(base.is_dir() and not base.is_symlink(), 'Missing or unsafe reviewed vendor source')
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(base)
            if any(p.startswith('.') or p in DENIED for p in relative.parts):
                continue
            if path.is_file() and (path.suffix in TEXT_SUFFIXES or 'license' in path.name.lower()):
                add(path, 'vendor/' + vendor + '/' + str(relative))
    apache = ROOT / '.venv/lib/python3.11/site-packages/diffusers-0.41.0.dist-info/licenses/LICENSE'
    add(apache, 'licenses/Diffusers-Apache-2.0-LICENSE.txt')
    add(apache, 'vendor/passt_hear21/LICENSE')
    add(args.protocol, 'analysis/experiment_protocol_frozen.json')
    add(args.report, STUDENT + '_report.pdf')
    add(args.report.with_suffix('.checks.json'), STUDENT + '_report.checks.json')
    for name, directory in directories.items():
        add(directory / 'manifest.json', f'results/{name}/manifest.json')
        for i in range(100 if name == 'baseline' else 20):
            add(directory / f'generated_samples/{i}.mp3', f'results/{name}/generated_samples/{i}.mp3')
    add(args.evaluation, 'results/baseline_evaluation.json')
    # The working top-k audit contains every original test caption. Keep it
    # local rather than republishing the released caption dataset.
    for name in ('retrieval.json', 'retrieval_manifest.json'):
        add(args.retrieval_dir / name, 'results/retrieval/' + name)
    if not args.baseline_only:
        for name in ('ablation_results.json', 'ablation_per_caption.csv'):
            add(args.ablation_dir / name, 'results/ablation/' + name)
        report_checks = read(args.report.with_suffix('.checks.json')) if args.report.with_suffix('.checks.json').is_file() else {}
        for key in ('tradeoff_figure_png', 'tradeoff_figure_pdf'):
            entry = report_checks.get('inputs', {}).get(key)
            if entry:
                source = Path(entry['path'])
                check(sha(safe_source(source)) == entry['sha256'], 'Report figure changed after validation')
                add(source, 'figures/' + source.name)
    if not args.draft:
        expected_clips = 100 if args.baseline_only else 160
        check(sum(n.endswith('.mp3') for n in selected) == expected_clips,
              'The public evidence must contain exactly ' + str(expected_clips) + ' MP3 clips')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hw3-source-', dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        for name, source in selected.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            check(sha(target) == sha(source), 'Source changed during copy: ' + name)
        (stage / 'README.md').write_text(public_readme(safe_source(ROOT / 'README.md').read_text(), args.draft,
                                                    args.baseline_only))
        notices = ('Reviewed upstream source: av-benchmark f351b9a6fc6abde746d5f8e1d4c47c883319cb41 (MIT), '
                   'passt_hear21 1ab778984f67cca09dd8bbad11f728a0122cce24 (Apache-2.0, declared in its setup.py).\n'
                   'Diffusers 0.41.0 adapted sampling portions: Apache-2.0. Full license is included.\n'
                   'The PaSST repository declares Apache-2.0 but lacks a root LICENSE; the standard full '
                   'Apache-2.0 text is supplied from the installed Diffusers distribution. '
                   'Existing upstream notices are preserved. No pretrained weights are redistributed.\n')
        (stage / 'THIRD_PARTY_NOTICES.txt').write_text(notices)
        model_dir = stage / 'model_provenance'; model_dir.mkdir()
        if args.model_provenance.is_file():
            model = read(safe_source(args.model_provenance))
            public = {k: model[k] for k in ('repository', 'revision', 'completed', 'files', 'total_bytes') if k in model}
        else:
            public = {'repository': 'stabilityai/stable-audio-open-1.0', 'revision': SAO_REVISION,
                      'completed': False, 'status': 'DRAFT: complete verified model provenance is not available'}
            missing.append('model_provenance/stable_audio_open.json (complete official file hashes)')
        (model_dir / 'stable_audio_open.json').write_text(json.dumps(public, indent=2) + '\n')
        clap = {'repository': 'lukewys/laion_clap', 'revision': 'b3708341862f581175dba5c356a4ebf74a9b6651',
                'filename': 'music_audioset_epoch_15_esc_90.14.pt', 'sha256': CLAP_SHA,
                'architecture': 'HTSAT-base', 'enable_fusion': False}
        (model_dir / 'clap.json').write_text(json.dumps(clap, indent=2) + '\n')
        encoder_probe = ROOT / 'analysis/evaluator_encoder_smoke.json'
        if encoder_probe.is_file():
            probe = read(safe_source(encoder_probe))
            encoders = {k: v['provenance'] for k, v in probe.get('encoders', {}).items()
                        if v.get('passed') is True and isinstance(v.get('provenance'), dict)}
            (model_dir / 'evaluation_encoders.json').write_text(json.dumps({
                'source': 'Verified pretrained encoder load records; not submitted music or metric results',
                'complete': probe.get('status') == 'passed', 'encoders': encoders}, indent=2) + '\n')
        manifest = {'status': 'DRAFT_INCOMPLETE_NOT_FOR_SUBMISSION' if args.draft else 'FORMAL_VALIDATED_LOCAL_BUNDLE',
                    'scope': scope,
                    'created_at_utc': datetime.now(timezone.utc).isoformat(), 'student_id': STUDENT,
                    'published': False, 'contains_released_dataset': False, 'contains_model_weights': False,
                    'contains_credentials': False, 'no_self_trained_checkpoint': True,
                    'validation': validation, 'missing_required_artifacts': missing, 'files': {}}
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                name = str(path.relative_to(stage)); safe_member(name)
                check(not path.is_symlink() and path.stat().st_size <= MAX_FILE, 'Unsafe or oversized staged file')
                if path.suffix in TEXT_SUFFIXES and re.search(rb'hf_[A-Za-z0-9]{24,}', path.read_bytes()):
                    raise ValueError('Possible credential detected in a selected text file; nothing was published')
                manifest['files'][name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
        total = sum(r['bytes'] for r in manifest['files'].values())
        check(total <= MAX_TOTAL, 'Public source bundle exceeds the 100 MiB size limit')
        (stage / 'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        expected = {output.name + '/' + n: row['sha256'] for n, row in manifest['files'].items()}
        expected[output.name + '/BUNDLE_MANIFEST.json'] = sha(stage / 'BUNDLE_MANIFEST.json')
        working_zip = Path(temporary) / 'bundle.zip'
        with zipfile.ZipFile(working_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for member in sorted(expected):
                archive.write(stage / Path(member).relative_to(output.name), member)
        verify_zip(working_zip, expected)
        zip_hash = sha(working_zip)
        stage.replace(output)
        working_zip.replace(output.with_suffix('.zip'))
    return {'status': manifest['status'], 'scope': scope,
            'folder': str(output), 'archive': str(output.with_suffix('.zip')),
            'archive_sha256': zip_hash, 'files': len(expected), 'uncompressed_bytes': total,
            'archive_bytes': output.with_suffix('.zip').stat().st_size,
            'missing_required_artifacts': missing, 'zip_crc_and_all_hashes_verified': True,
            'published': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--draft', action='store_true', help='Create a clearly labeled incomplete preview only')
    p.add_argument('--baseline-only', action='store_true',
                   help='Validate/package only the full 100-clip baseline, required metrics and matching baseline-only report')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--generation-dir', type=Path, default=ROOT / 'results/baseline')
    p.add_argument('--condition', action='append', default=[], metavar='NAME=PATH')
    p.add_argument('--protocol', type=Path, default=ROOT / 'analysis/experiment_protocol_frozen.json')
    p.add_argument('--report', type=Path, default=ROOT / (STUDENT + '_report.pdf'))
    p.add_argument('--evaluation', type=Path, default=ROOT / 'results/baseline_evaluation.json')
    p.add_argument('--retrieval-dir', type=Path, default=ROOT / 'results/retrieval')
    p.add_argument('--ablation-dir', type=Path, default=ROOT / 'results/ablation')
    p.add_argument('--cool-zip', type=Path, default=ROOT / (STUDENT + '.zip'))
    p.add_argument('--model-provenance', type=Path,
                   default=ROOT / 'cache/models/stable-audio-open-1.0-f21265c1/model_provenance.json')
    args = p.parse_args()
    try:
        result = package(args)
        missing = result.pop('missing_required_artifacts')
        result.update(missing_required_artifacts_count=len(missing),
                      full_missing_list='BUNDLE_MANIFEST.json inside the generated folder/archive')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        p.exit(1, 'Source bundle not ready: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
