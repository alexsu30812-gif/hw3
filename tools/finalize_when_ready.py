"""Wait for genuine HW3 results, finalize local artifacts, optionally publish.

--plan performs no writes, waits, subprocesses or network requests. Normal runs
wait at most 18 hours and never publish unless --publish-github is explicit.
Existing repositories/results are preserved; no force push or remote deletion
is supported. Authentication is delegated to Git's existing credential helper.
--baseline-only validates the real required artifacts immediately, without
waiting for extra experiments, and preserves an immutable COOL-ready snapshot.
The later full publication may only fast-forward this finalizer's own recorded
baseline commit, after checking the remote, clean clone and every prior file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import re
import shutil
import shlex
import ssl
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
READY = 'local_experiments_and_cool_zip_complete_report_is_draft'
BASELINE_SCOPE = 'baseline_only'
FULL_SCOPE = 'full_with_ablation'
PROGRESS_SCOPE = 'work_in_progress'
REPOSITORY = 'https://github.com/alexsu30812-gif/hw3.git'
PUBLIC_URL = 'https://github.com/alexsu30812-gif/hw3'
SLUG = 'alexsu30812-gif/hw3'
STUDENT = 'r14942154'
CONDITIONS = ('baseline', 'schedule_only', 'rescale_only', 'combined')
METRICS = ('FD-VGG-all-reference', 'FD-VGG-retrieved-proxy',
           'KL-PASST-softmax-retrieved-proxy', 'CLAP')
DENIED_PARTS = {'.git', '.venv', 'cache', 'caches', 'audio_mp3', 'data',
                'weights', '__pycache__', '.aws', '.codex', '.agents', 'stored_tokens'}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def local(path, required=False):
    """Keep every input/output below HW3, rejecting symlinks and '..'."""
    path = Path(path).absolute()
    check(path.is_relative_to(ROOT), 'Path must be within the HW3 directory')
    cursor = ROOT
    for part in path.relative_to(ROOT).parts:
        check(part not in {'.', '..'}, 'Escaping path rejected')
        cursor /= part
        check(not cursor.is_symlink(), 'Symlink input/output rejected: ' + str(path.relative_to(ROOT)))
    check(path.resolve().is_relative_to(ROOT.resolve()), 'Resolved path escapes HW3')
    if required:
        check(path.is_file(), 'Required local file is missing: ' + str(path.relative_to(ROOT)))
    return path


def read(path):
    return json.loads(local(path, required=True).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    check(not temporary.is_symlink(), 'Unsafe temporary metadata path')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def redact(value):
    value = re.sub(r'hf_[A-Za-z0-9]{20,}', '[REDACTED]', str(value))
    value = re.sub(r'gh[pousr]_[A-Za-z0-9_]{20,}', '[REDACTED]', value)
    value = re.sub(r'(https?://)[^/\s@]+@', r'\1[REDACTED]@', value)
    value = re.sub(r'(?i)(authorization:\s*(?:bearer|basic)\s+)\S+', r'\1[REDACTED]', value)
    return value


class Recorder:
    def __init__(self, args):
        self.path = args.state
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        self.directory = local(ROOT / 'analysis/finalizer_logs' / stamp)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.value = {'status': 'starting', 'started_at_utc': now(),
                      'scope': BASELINE_SCOPE if args.baseline_only else FULL_SCOPE,
                      'publish_github_requested': args.publish_github,
                      'published': False, 'steps': {}, 'finalizer_sha256': sha(__file__)}
        if self.path.exists():
            previous = read(self.path)
            atomic_json(self.directory / 'previous_state.json', previous)
        self.save()

    def save(self):
        self.value['updated_at_utc'] = now()
        atomic_json(self.path, self.value)

    def status(self, name, **values):
        self.value.update(status=name, **values)
        self.save()
        print(json.dumps({'status': name, **values}, ensure_ascii=False), flush=True)

    def step(self, name, action, inputs=()):
        row = {'status': 'running', 'started_at_utc': now(),
               'input_sha256': {str(local(p, required=True).relative_to(ROOT)): sha(p) for p in inputs}}
        self.value['steps'][name] = row
        self.save()
        try:
            result = action()
            row.update(status='complete', completed_at_utc=now(), result=result)
            self.save()
            return result
        except Exception as error:
            row.update(status='failed', completed_at_utc=now(), error=redact(error))
            self.save()
            raise

    def command(self, name, command, timeout=900, okay=(0,), cwd=ROOT):
        if self.value.get('background_cloud_retry') and str(command[0]) == 'git':
            timeout = min(timeout, 120)
        log = self.directory / (name + '.log')
        environment = dict(os.environ)
        environment.update(GIT_TERMINAL_PROMPT='0', GIT_TRACE='0', GIT_CURL_VERBOSE='0',
                           GIT_TRACE_CURL='0', LC_ALL='C')
        if self.value.get('background_cloud_retry'):
            environment['GCM_INTERACTIVE'] = 'never'
        for key in ('GIT_CURL_VERBOSE', 'GIT_TRACE_CURL', 'GIT_TRACE_CURL_NO_DATA'):
            environment.pop(key, None)
        try:
            result = subprocess.run([str(p) for p in command], cwd=cwd, env=environment,
                                    text=True, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            def decoded(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value or ''
            log.write_text(redact('$ ' + ' '.join(str(x) for x in command) + '\n'
                                  + decoded(error.stdout) + '\n' + decoded(error.stderr) + '\nTIMED OUT\n'))
            self.value.setdefault('commands', {})[name] = {
                'command': [str(p) for p in command], 'exit_code': None, 'timed_out': True,
                'log': str(log.relative_to(ROOT)), 'log_sha256': sha(log)}
            self.save()
            raise
        contents = '$ ' + ' '.join(str(x) for x in command) + '\n' + result.stdout + '\n' + result.stderr
        log.write_text(redact(contents))
        record = {'command': [str(p) for p in command], 'exit_code': result.returncode,
                  'log': str(log.relative_to(ROOT)), 'log_sha256': sha(log)}
        self.value.setdefault('commands', {})[name] = record
        self.save()
        check(result.returncode in okay,
              name + ' failed with exit ' + str(result.returncode) + '; see ' + str(log.relative_to(ROOT)))
        return result.stdout


def destination(path):
    value = read(path)
    check(value.get('repository') == REPOSITORY and value.get('report_url') == PUBLIC_URL,
          'The destination must be the HW3 repository explicitly provided by the user')
    return value


def configure_task_git_auth(recorder, clone):
    """Use the user's browser-authorized GCM account only in our task clone."""
    config = ROOT / 'analysis/git_auth_config.json'
    if not config.is_file():
        return
    saved = read(config)
    helper = '/Applications/GitHub Desktop.app/Contents/Resources/app/git/libexec/git-core/git-credential-manager'
    check(saved == {'provider': 'git_credential_manager', 'helper': helper, 'username': 'alexsu30812-gif'}
          and Path(helper).is_file(), 'Unexpected task Git authentication configuration')
    recorder.command('task_auth_reset', ['git', '-C', clone, 'config', '--local', '--replace-all', 'credential.helper', ''])
    recorder.command('task_auth_helper', ['git', '-C', clone, 'config', '--local', '--add',
                                        'credential.helper', '!' + shlex.quote(helper)])
    recorder.command('task_auth_account', ['git', '-C', clone, 'config', '--local', 'credential.username', saved['username']])


def wait_for_pipeline(args, recorder):
    deadline = time.monotonic() + args.max_wait_hours * 3600
    previous = None
    incomplete_reads = 0
    while True:
        if args.pipeline_state.exists():
            try:
                state = read(args.pipeline_state)
                incomplete_reads = 0
            except json.JSONDecodeError:
                incomplete_reads += 1
                check(incomplete_reads <= 3, 'Pipeline state remains invalid JSON; preserve outputs and stop')
                check(time.monotonic() < deadline, 'Timed out while pipeline state was incomplete')
                time.sleep(min(args.poll_seconds, max(0.01, deadline - time.monotonic())))
                continue
            status = state.get('status', '')
            if status != previous:
                recorder.status('waiting_for_pipeline', pipeline_status=status,
                                pipeline_state_sha256=sha(args.pipeline_state))
                previous = status
            if status == READY:
                return {'pipeline_status': status, 'pipeline_state_sha256': sha(args.pipeline_state)}
            check(not any(term in str(status).lower() for term in ('stop', 'fail', 'error', 'abort', 'blocked', 'cancel')),
                  'The pipeline stopped: ' + str(status) + '. No finalization or publication was performed.')
        check(time.monotonic() < deadline, 'Timed out waiting for real pipeline completion; no final result is claimed')
        time.sleep(min(args.poll_seconds, max(0.01, deadline - time.monotonic())))


def validate_results(baseline_only=False):
    evaluation = read(ROOT / 'results/baseline_evaluation.json')
    ablation = None if baseline_only else read(ROOT / 'results/ablation/ablation_results.json')
    check(evaluation.get('completed') is True and evaluation.get('counts', {}).get('generated_clips') == 100
          and evaluation['counts'].get('fd_reference_clips') == 5367
          and evaluation['counts'].get('paired_reference_rows') == 100, 'Required real full-100 metrics are incomplete')
    for name in METRICS:
        value = evaluation.get(name)
        check(type(value) in (int, float) and math.isfinite(value) and (name == 'CLAP' or value >= -1e-6),
              'Missing/invalid real metric: ' + name)
    if not baseline_only:
        cohort = [str(i) for i in range(20)]
        cells = {(r['condition'], r['id']) for r in ablation.get('per_caption', [])}
        check(ablation.get('completed') is True and ablation.get('n_conditions') == 4
              and ablation.get('n_captions') == 20 and ablation.get('cohort_ids') == cohort
              and len(ablation['per_caption']) == 80 and cells == {(n, i) for n in CONDITIONS for i in cohort},
              'The genuine four-condition paired ablation is incomplete')
    hashes = {}
    for name in ('baseline',) if baseline_only else CONDITIONS:
        directory = ROOT / 'results' / name
        manifest = read(directory / 'manifest.json')
        rows = manifest.get('samples', [])
        expected = {str(i) for i in range(100 if name == 'baseline' else 20)}
        check(len(rows) == len(expected) and {r['id'] for r in rows} == expected,
              'Incomplete or duplicated actual generated IDs: ' + name)
        if name == 'baseline':
            check(manifest.get('complete') is True and manifest.get('formal_complete') is True,
                  'Main generation is not a complete formal run')
        for row in rows:
            check(row['mp3']['path'] == f"generated_samples/{row['id']}.mp3", 'Unexpected MP3 path')
            path = local(directory / row['mp3']['path'], required=True)
            check(sha(path) == row['mp3']['sha256'] and row.get('audio_audit', {}).get('codec') == 'mp3',
                  'An actual MP3 changed or lacks recorded decode evidence')
            if name == 'baseline':
                hashes[f"{STUDENT}/generated_samples/{row['id']}.mp3"] = sha(path)
    from tools.package_source import verify_zip
    hashes[f'{STUDENT}/cfg.py'] = sha(ROOT / 'cfg.py')
    hashes[f'{STUDENT}/retrieval.json'] = sha(ROOT / 'results/retrieval/retrieval.json')
    verify_zip(local(ROOT / (STUDENT + '.zip'), required=True), hashes)
    return evaluation, ablation


def update_readme(evaluation, artifact_url, baseline_only=False):
    path = local(ROOT / 'README.md', required=True)
    before = sha(path)
    original = path.read_text()
    check(len(re.findall(r'^## Current status\s*$', original, flags=re.M)) == 1,
          'Expected exactly one Current status section; preserve README for review')
    rows = '\n'.join(f'| {name} | {evaluation[name]:.6f} |' for name in METRICS)
    completed = ('''The required baseline artifacts are complete: **100 genuine baseline music
clips**, CLAP retrieval for all 100 captions against 5367 MusicCaps references,
and the required full-100 evaluation. The baseline was selected before test
scores. This **baseline-only submission** does not claim completed extra CFG
experiments or ablation results. Additional experiments are kept separate and
may produce a later report revision after their actual completion.'''
        if baseline_only else '''The recorded local experiments are complete: **100 baseline music clips**,
**20 clips for each of three extra CFG conditions** (160 MP3 files total),
CLAP retrieval for all 100 captions against 5367 MusicCaps references, and
the required full-100 evaluation. The paired 2×2 ablation covers the same
predeclared 20 caption IDs in four conditions; it does not replace the full
100-clip evaluation. The baseline was selected before test scores.''')
    replacement = f'''## Current status

{completed}

| Required evaluation / explicitly labeled reference policy | Recorded score |
| --- | ---: |
{rows}

Scores are read from the completed experiment JSON, not estimates. FD against
all 5367 references and FD against 100 retrieved rows are different populations.
KL is **KL(retrieved reference || generated)**: retrieved audio is a proxy,
not ground truth for the new caption. Repeated retrieved references are counted
as paired rows; the unique count is reported in the evaluation JSON. The TA's
intended reference policy remains unconfirmed. CLAP measures global alignment,
and none of these metrics establishes human musical preference. **MQAScore was
not implemented or claimed.** No human listening is implied by signal metrics.

Public artifact destination: {artifact_url}
GitHub substitution for the requested cloud-drive folder remains unconfirmed
by the TA. Local finalization and publication verification are separately
recorded in `analysis/finalization_state.json`; this destination alone does not
claim that upload has succeeded.

The course deadline is **2026-11-16 23:59 GMT+8** (2026-11-16 08:59 Phoenix).
The late acceptance window is not an extension.

'''
    updated = re.sub(r'^## Current status\s*\n.*?(?=^## |\Z)', lambda _: replacement,
                     original, count=1, flags=re.M | re.S)
    temporary = path.with_name('README.md.finalizing.tmp')
    check(not temporary.is_symlink(), 'Unsafe README temporary path')
    temporary.write_text(updated)
    temporary.replace(path)
    return {'before_sha256': before, 'after_sha256': sha(path), 'scores': {n: evaluation[n] for n in METRICS}}


def tls_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def public_repo(recorder, label):
    request = Request('https://api.github.com/repos/' + SLUG,
                      headers={'User-Agent': 'HW3-local-artifact-finalizer', 'Accept': 'application/vnd.github+json'})
    with urlopen(request, timeout=45, context=tls_context()) as response:
        check(response.status == 200, 'Unauthenticated GitHub repository lookup did not succeed')
        data = json.loads(response.read(1024 * 1024))
    check(data.get('full_name') == SLUG and data.get('private') is False
          and data.get('visibility', 'public') == 'public' and not data.get('archived'),
          'The explicitly selected repository must be public and writable (not archived)')
    public = {k: data.get(k) for k in ('id', 'full_name', 'private', 'visibility', 'default_branch', 'archived')}
    path = recorder.directory / (label + '.json')
    atomic_json(path, public)
    return {'unauthenticated_public_check': True, 'record_sha256': sha(path), **public}


def empty_remote(recorder, label):
    result = recorder.command(label, ['git', 'ls-remote', REPOSITORY], timeout=120)
    check(not result.strip(), 'Remote now contains refs. Preserve local artifacts and stop; no overwrite is allowed.')
    return {'remote_empty': True}


def public_member(name, path):
    relative = Path(name)
    check(not relative.is_absolute() and '..' not in relative.parts
          and '\n' not in name and '\r' not in name
          and not any(p.startswith('.') or p in DENIED_PARTS for p in relative.parts), 'Unsafe publication member')
    check(relative.suffix.lower() not in {'.pt', '.pth', '.safetensors', '.bin', '.ckpt', '.wav'},
          'Model weights or working WAV masters cannot be published')
    check(name not in {'test.csv', 'sample_code/test.csv'} and 'topk_and_tokenization' not in name,
          'Released captions/detailed top-k captions cannot be published')
    check(not path.is_symlink() and path.is_file() and path.stat().st_size < 100 * 1024 * 1024,
          'Missing, symlinked or GitHub-oversized publication file')


def validate_formal_report(report, scope):
    checks = read(report.with_suffix('.checks.json'))
    check(checks.get('draft') is False and checks.get('formal_result_validation_passed') is True
          and checks.get('report_scope') == scope
          and checks.get('sha256') == sha(local(report, required=True)),
          'The generated PDF failed formal validation or has the wrong report scope')
    return checks


def freeze_cool_ready(report, source):
    """Preserve the first validated baseline PDF/ZIP as a fixed upload copy."""
    validate_formal_report(report, BASELINE_SCOPE)
    directory = local(ROOT / 'deliverables/cool_ready')
    metadata = directory / 'COOL_READY_MANIFEST.json'
    paths = {p.name: local(p, required=True) for p in
             (report, report.with_suffix('.checks.json'), ROOT / (STUDENT + '.zip'),
              ROOT / (STUDENT + '.checks.json'))}
    evidence = {'scope': BASELINE_SCOPE,
                'protocol_sha256': sha(local(ROOT / 'analysis/experiment_protocol_frozen.json', required=True)),
                'evaluation_sha256': sha(local(ROOT / 'results/baseline_evaluation.json', required=True)),
                'cool_zip_sha256': sha(paths[STUDENT + '.zip'])}
    if directory.exists():
        saved = read(metadata)
        check(saved.get('status') == 'FORMAL_BASELINE_COOL_READY'
              and saved.get('evidence') == evidence,
              'Existing fixed COOL snapshot has other inputs; preserve it')
        actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
        check(actual == set(paths) | {'COOL_READY_MANIFEST.json'}, 'Fixed COOL folder contains unexpected files')
        for name, info in saved['files'].items():
            path = local(directory / name, required=True)
            check(sha(path) == info['sha256'] and path.stat().st_size == info['bytes'],
                  'The fixed COOL-ready copy changed after preservation')
        validate_formal_report(directory / report.name, BASELINE_SCOPE)
        return {'directory': str(directory.relative_to(ROOT)), 'manifest_sha256': sha(metadata),
                'existing_fixed_copy_preserved': True}
    directory.mkdir(parents=True, exist_ok=False)
    copied = {}
    for name, path in paths.items():
        target = local(directory / name)
        shutil.copyfile(path, target)
        check(sha(target) == sha(path), 'Fixed COOL upload copy differs from its validated source')
        copied[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
    atomic_json(metadata, {'status': 'FORMAL_BASELINE_COOL_READY', 'evidence': evidence,
                          'files': copied, 'created_at_utc': now(),
                          'source_bundle_manifest_sha256': sha(local(source / 'BUNDLE_MANIFEST.json', required=True)),
                          'immutable_baseline_snapshot': True})
    return {'directory': str(directory.relative_to(ROOT)), 'manifest_sha256': sha(metadata),
            'existing_fixed_copy_preserved': False}


def previous_baseline_publication(selected, scope):
    """Only our verified progress/baseline publication authorizes an update."""
    if selected.get('publication_scope') == PROGRESS_SCOPE:
        saved = selected.get('progress_publication', {})
        check(scope in {BASELINE_SCOPE, FULL_SCOPE}
              and selected.get('publishing_completed') is False
              and saved.get('scope') == PROGRESS_SCOPE
              and saved.get('remote_and_public_snapshot_verified') is True
              and saved.get('commit') == selected.get('commit')
              and saved.get('tree') == selected.get('publication_tree')
              and saved.get('files_sha256') == selected.get('publication_files_sha256'),
              'Only the recorded verified progress publication may receive the formal update')
        check(re.fullmatch(r'[0-9a-f]{40,64}', saved.get('commit', ''))
              and re.fullmatch(r'[0-9a-f]{40,64}', saved.get('tree', ''))
              and isinstance(saved.get('files_sha256'), dict)
              and {'README.md', 'PROGRESS_SNAPSHOT.json'}.issubset(saved['files_sha256'])
              and all(isinstance(n, str) and isinstance(h, str) and re.fullmatch(r'[0-9a-f]{64}', h)
                      for n, h in saved['files_sha256'].items()),
              'Saved progress publication inventory is invalid')
        return saved
    if not selected.get('publishing_completed') and not selected.get('commit') and not selected.get('baseline_publication'):
        return None
    saved = selected.get('baseline_publication', {})
    check(scope == FULL_SCOPE and selected.get('publishing_completed') is True
          and selected.get('publication_scope') == BASELINE_SCOPE
          and selected.get('report_and_zip_public_sha_verified') is True
          and saved.get('scope') == BASELINE_SCOPE and saved.get('commit') == selected.get('commit')
          and saved.get('files_sha256') == selected.get('publication_files_sha256'),
          'Only the finalizer\'s own verified prior baseline commit may receive the full update')
    check(re.fullmatch(r'[0-9a-f]{40,64}', saved.get('commit', ''))
          and re.fullmatch(r'[0-9a-f]{40,64}', saved.get('tree', ''))
          and isinstance(saved.get('files_sha256'), dict) and saved['files_sha256'],
          'Saved baseline commit/tree/file ownership is incomplete')
    for name, digest in saved['files_sha256'].items():
        check(isinstance(name, str) and isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest),
              'Saved baseline publication inventory is invalid')
    check({STUDENT + '_report.pdf', STUDENT + '.zip'}.issubset(saved['files_sha256']),
          'Saved baseline publication lacks its verified report or COOL ZIP')
    return saved


def expected_remote(recorder, label, commit):
    refs = recorder.command(label, ['git', 'ls-remote', REPOSITORY, 'HEAD', 'refs/heads/main'], timeout=120)
    remote = {line.split()[1]: line.split()[0] for line in refs.splitlines() if line.strip()}
    check(remote.get('HEAD') == commit and remote.get('refs/heads/main') == commit,
          'Remote HEAD/main changed from our prior baseline commit; third-party changes will not be overwritten')
    return {'remote_head': commit, 'remote_main': commit}


def verify_owned_tree(recorder, clone, commit, files, tree=None, label='owned_clone'):
    head = recorder.command(label + '_head', ['git', '-C', clone, 'rev-parse', 'HEAD']).strip()
    check(head == commit, 'Isolated clone HEAD differs from the recorded publication')
    check(not recorder.command(label + '_status', ['git', '-C', clone, 'status', '--porcelain=v1',
                                                   '--untracked-files=all']).strip(),
          'Isolated clone contains changes; preserve them instead of overwriting')
    actual = set()
    for path in clone.rglob('*'):
        relative = path.relative_to(clone)
        if relative.parts[0] == '.git':
            continue
        check(not path.is_symlink(), 'Isolated clone contains an unexpected symlink')
        if path.is_file():
            actual.add(str(relative))
    check(actual == set(files), 'Isolated clone contains files outside its prior published whitelist')
    names = recorder.command(label + '_names', ['git', '-C', clone, 'ls-tree', '-r', '--name-only', '-z', 'HEAD'])
    check(set(filter(None, names.split('\0'))) == set(files), 'Committed tree differs from the recorded file whitelist')
    for name, digest in files.items():
        path = local(clone / name, required=True)
        public_member(name, path)
        check(sha(path) == digest, 'A prior published file changed locally: ' + name)
    actual_tree = recorder.command(label + '_tree', ['git', '-C', clone, 'rev-parse', 'HEAD^{tree}']).strip()
    check(tree is None or actual_tree == tree, 'Isolated clone tree differs from the recorded baseline tree')
    return actual_tree


def copy_public_bundle(source, clone, prior_files=None, expected_scope=FULL_SCOPE):
    prior_files = {} if prior_files is None else prior_files
    manifest = read(source / 'BUNDLE_MANIFEST.json')
    check(manifest.get('status') == 'FORMAL_VALIDATED_LOCAL_BUNDLE'
          and manifest.get('scope') == expected_scope
          and manifest.get('validation', {}).get('scope') == expected_scope
          and manifest.get('validation', {}).get('validated') is True
          and not manifest.get('missing_required_artifacts'), 'Only the fully validated formal source bundle can be published')
    report = local(source / (STUDENT + '_report.pdf'), required=True)
    validate_formal_report(report, expected_scope)
    check(manifest['validation'].get('report_sha256') == sha(report),
          'Source validation refers to a different formal report')
    files = {name: local(source / name, required=True) for name in manifest['files']}
    files['BUNDLE_MANIFEST.json'] = local(source / 'BUNDLE_MANIFEST.json', required=True)
    actual_files = {str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    check(actual_files == set(files), 'Source folder has unexpected files outside its verified manifest')
    for name, record in manifest['files'].items():
        check(sha(files[name]) == record['sha256'] and files[name].stat().st_size == record['bytes'],
              'Source bundle changed after packaging: ' + name)
    files[STUDENT + '.zip'] = local(ROOT / (STUDENT + '.zip'), required=True)
    files[STUDENT + '.checks.json'] = local(ROOT / (STUDENT + '.checks.json'), required=True)
    check(manifest['validation'].get('cool_zip_sha256') == sha(files[STUDENT + '.zip']),
          'COOL ZIP changed after source bundle validation')
    copied = {}
    for name, path in sorted(files.items()):
        public_member(name, path)
        target = local(clone / name)
        if target.exists():
            check(name in prior_files and target.is_file() and sha(target) == prior_files[name],
                  'Publication target is not an unchanged prior authorized file; preserve it')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != sha(path):
            temporary = local(target.with_name(target.name + '.publishing.tmp'))
            check(not temporary.exists(), 'Unexpected publication temporary file; preserve it')
            shutil.copyfile(path, temporary)
            temporary.replace(target)
        check(sha(target) == sha(path), 'Public copy hash mismatch')
        copied[name] = sha(target)
    return copied


def verify_raw_public(commit, files, max_attempts=8, timeout=60):
    verified = {}
    for name, expected in files.items():
        url = f'https://raw.githubusercontent.com/{SLUG}/{commit}/{name}'
        last = None
        for attempt in range(max_attempts):
            try:
                request = Request(url, headers={'User-Agent': 'HW3-local-artifact-finalizer'})
                hasher, size = hashlib.sha256(), 0
                with urlopen(request, timeout=timeout, context=tls_context()) as response:
                    check(response.status == 200, 'Public raw artifact is not HTTP 200')
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        check(size <= expected['bytes'] + 1024, 'Unexpected public artifact size')
                        hasher.update(chunk)
                check(hasher.hexdigest() == expected['sha256'] and size == expected['bytes'],
                      'Public raw artifact bytes differ from the actual submission')
                verified[name] = {'url': url, 'sha256': hasher.hexdigest(), 'bytes': size,
                                  'unauthenticated_http_200': True}
                break
            except (HTTPError, URLError, TimeoutError, ValueError) as error:
                last = error
                if attempt < max_attempts - 1:
                    time.sleep(5)
        check(name in verified, 'Public raw verification failed after bounded retries: ' + name + ' (' + redact(last) + ')')
    return verified


def publish(args, recorder, source):
    scope = BASELINE_SCOPE if args.baseline_only else FULL_SCOPE
    selected = destination(args.destination)
    prior = previous_baseline_publication(selected, scope)
    prior_files = {} if prior is None else prior['files_sha256']
    recorder.step('public_repo_before_clone', lambda: public_repo(recorder, 'repo_public_before'))
    if prior is None:
        recorder.step('empty_remote_before_clone', lambda: empty_remote(recorder, 'remote_before_clone'))
    else:
        recorder.step('owned_baseline_remote_before_clone',
                      lambda: expected_remote(recorder, 'remote_before_clone', prior['commit']))
    clone = local(ROOT / 'github_publish')
    if clone.exists():
        check(clone.is_dir() and (clone / '.git').is_dir() and not (clone / '.git').is_symlink(),
              'Existing github_publish is not the isolated clone; preserve it')
        if prior is None:
            check({p.name for p in clone.iterdir()}.issubset({'.git'}),
                  'Existing github_publish contains work; preserve it and stop')
            head = recorder.command('existing_clone_head', ['git', '-C', clone, 'rev-parse', '--verify', 'HEAD'], okay=(0, 128)).strip()
            check(not head, 'Existing isolated clone already has an unrecorded commit; preserve it')
    else:
        recorder.command('isolated_clone', ['git', '-c', 'protocol.file.allow=never', 'clone', '--', REPOSITORY, clone], timeout=300)
    configure_task_git_auth(recorder, clone)
    origin = recorder.command('isolated_clone_origin', ['git', '-C', clone, 'remote', 'get-url', 'origin']).strip()
    push_origin = recorder.command('isolated_clone_push_origin', ['git', '-C', clone, 'remote', 'get-url', '--push', 'origin']).strip()
    check(origin == REPOSITORY and push_origin == REPOSITORY, 'Isolated clone has a different fetch/push destination')
    top = recorder.command('clone_root', ['git', '-C', clone, 'rev-parse', '--show-toplevel']).strip()
    check(Path(top).resolve() == clone.resolve(), 'Git operation escaped the isolated HW3 clone')
    if prior is not None:
        recorder.step('verify_owned_baseline_clone', lambda: verify_owned_tree(
            recorder, clone, prior['commit'], prior_files, prior['tree'], 'baseline_clone'))
    copied = recorder.step('copy_verified_public_files', lambda: copy_public_bundle(source, clone, prior_files, scope),
                           inputs=(source / 'BUNDLE_MANIFEST.json',))
    if prior is None:
        recorder.command('set_unborn_main', ['git', '-C', clone, 'symbolic-ref', 'HEAD', 'refs/heads/main'])
    else:
        branch = recorder.command('owned_baseline_branch', ['git', '-C', clone, 'symbolic-ref', '--short', 'HEAD']).strip()
        check(branch == 'main', 'Owned baseline clone is not on main; preserve its branch')
    for key, fallback in (('user.name', STUDENT), ('user.email', 'alexsu30812-gif@users.noreply.github.com')):
        value = recorder.command('read_' + key.replace('.', '_'), ['git', '-C', clone, 'config', '--get', key], okay=(0, 1)).strip()
        if not value:
            recorder.command('set_local_' + key.replace('.', '_'), ['git', '-C', clone, 'config', '--local', key, fallback])
    recorder.command('stage_whitelist', ['git', '-C', clone, 'add', '--', *sorted(copied)])
    changed = {name for name, digest in copied.items() if prior_files.get(name) != digest}
    check(changed, 'The verified publication contains no new or changed authorized files')
    staged = recorder.command('staged_names', ['git', '-C', clone, 'diff', '--cached', '--name-only', '-z'])
    check(set(filter(None, staged.split('\0'))) == changed, 'Git index contains unexpected or missing changed files')
    status = recorder.command('staged_status', ['git', '-C', clone, 'status', '--porcelain=v1', '-z', '--untracked-files=all'])
    entries = [x for x in status.split('\0') if x]
    check(len(entries) == len(changed)
          and all(x[:3] == ('M  ' if x[3:] in prior_files else 'A  ') and x[3:] in changed for x in entries),
          'Git status includes changes/deletions outside the authorized publication whitelist')
    check(all(sha(clone / name) == digest for name, digest in copied.items()), 'Copied files changed before commit')
    message = ('Add validated HW3 baseline report, predictions and reproducible source' if args.baseline_only else
               'Add paired CFG ablation and update the validated HW3 report' if prior is not None else
               'Add completed HW3 experiments, report and reproducible source')
    recorder.command('commit', ['git', '-C', clone, 'commit', '-m', message], timeout=300)
    commit = recorder.command('local_commit', ['git', '-C', clone, 'rev-parse', 'HEAD']).strip()
    check(re.fullmatch(r'[0-9a-f]{40,64}', commit), 'Invalid local commit hash')
    if prior is not None:
        parent = recorder.command('full_update_parent', ['git', '-C', clone, 'rev-parse', 'HEAD^']).strip()
        check(parent == prior['commit'], 'Full update is not a direct child of our prior baseline commit')
    combined = {**prior_files, **copied}
    tree = verify_owned_tree(recorder, clone, commit, combined, label='committed_publication')
    recorder.value.update(local_commit=commit, publication_scope=scope,
                          prior_publication_commit=prior['commit'] if prior is not None else None,
                          prior_baseline_commit=prior['commit'] if prior is not None and prior.get('scope') == BASELINE_SCOPE else None,
                          publication_tree=tree, publication_files_sha256=combined)
    recorder.save()
    recorder.step('public_repo_before_push', lambda: public_repo(recorder, 'repo_public_before_push'))
    if prior is None:
        recorder.step('empty_remote_immediately_before_push', lambda: empty_remote(recorder, 'remote_before_push'))
    else:
        recorder.step('owned_baseline_remote_immediately_before_push',
                      lambda: expected_remote(recorder, 'remote_before_push', prior['commit']))
    recorder.status('publishing_verified_final_artifacts', push_attempted=True)
    recorder.command('push_main', ['git', '-C', clone, 'push', 'origin', 'HEAD:refs/heads/main'], timeout=3600)
    recorder.value['push_returned_success'] = True
    recorder.save()
    matched = False
    for attempt in range(6):
        refs = recorder.command('verify_remote_refs_' + str(attempt), ['git', 'ls-remote', REPOSITORY, 'HEAD', 'refs/heads/main'], timeout=120)
        remote = {line.split()[1]: line.split()[0] for line in refs.splitlines() if line.strip()}
        if remote.get('HEAD') == commit and remote.get('refs/heads/main') == commit:
            matched = True
            break
        time.sleep(5)
    check(matched, 'Push returned success, but remote HEAD/main could not both be verified; preserve all artifacts')
    artifacts = {name: {'sha256': combined[name], 'bytes': (clone / name).stat().st_size}
                 for name in (STUDENT + '_report.pdf', STUDENT + '.zip')}
    background = recorder.value.get('background_cloud_retry')
    public = recorder.step('verify_public_report_and_zip', lambda: verify_raw_public(
        commit, artifacts, max_attempts=1 if background else 8, timeout=15 if background else 60))
    check(destination(args.destination) == selected, 'Destination ownership record changed during publication; preserve all artifacts')
    event = {'scope': scope, 'commit': commit, 'tree': tree, 'files_sha256': combined,
             'published_at_utc': now(), 'report_and_zip_public_sha_verified': True}
    if scope == BASELINE_SCOPE:
        selected['baseline_publication'] = event
    selected.setdefault('publication_history', []).append(event)
    selected.update(publishing_completed=True, published_at_utc=now(), commit=commit,
                    publication_scope=scope, publication_tree=tree,
                    publication_files_sha256=combined,
                    prior_publication_commit=prior['commit'] if prior is not None else None,
                    prior_baseline_commit=prior['commit'] if prior is not None and prior.get('scope') == BASELINE_SCOPE else None,
                    report_and_zip_public_sha_verified=True)
    atomic_json(args.destination, selected)
    return {'commit': commit, 'scope': scope, 'repository': PUBLIC_URL, 'public_artifacts': public,
            'destination_record_sha256': sha(args.destination)}


def enqueue_publication(args, source):
    """Persist immutable, locally validated artifacts without any network call."""
    scope = BASELINE_SCOPE if args.baseline_only else FULL_SCOPE
    manifest_path = local(source / 'BUNDLE_MANIFEST.json', required=True)
    bundle = read(manifest_path)
    check(bundle.get('status') == 'FORMAL_VALIDATED_LOCAL_BUNDLE'
          and bundle.get('scope') == scope and bundle.get('validation', {}).get('validated') is True
          and bundle['validation'].get('scope') == scope, 'Only validated artifacts can enter the publication queue')
    validate_formal_report(source / (STUDENT + '_report.pdf'), scope)
    path = ROOT / 'analysis/formal_publication_queue.json'
    queue = read(path) if path.is_file() else {'history': []}
    ready = {'scope': scope, 'source': str(local(source).relative_to(ROOT)),
             'source_manifest_sha256': sha(manifest_path),
             'report_sha256': bundle['validation']['report_sha256'], 'ready_at_utc': now()}
    queue.update(status='pending_publication', latest_ready=ready, updated_at_utc=now())
    queue.setdefault('history', []).append({'event': 'local_artifacts_ready', **ready})
    atomic_json(path, queue)
    return ready


def resume_formal_commit(recorder, destination_path, pending):
    """Finish a checked formal commit after network loss, without rebuilding outputs."""
    selected = destination(destination_path)
    if selected.get('commit') == pending['commit'] and selected.get('publishing_completed') is True:
        return selected
    check(selected.get('commit') == pending.get('prior_commit'), 'Formal retry destination ownership changed')
    clone = local(ROOT / 'github_publish')
    verify_owned_tree(recorder, clone, pending['commit'], pending['files_sha256'], pending['tree'], 'queued_formal')
    for flag in ([], ['--push']):
        origin = recorder.command('queued_origin' + ('_push' if flag else ''),
            ['git', '-C', clone, 'remote', 'get-url', *flag, 'origin']).strip()
        check(origin == REPOSITORY, 'Queued clone has a different destination')
    refs = recorder.command('queued_remote', ['git', 'ls-remote', REPOSITORY], timeout=120)
    refmap = {line.split()[1]: line.split()[0] for line in refs.splitlines() if line.strip()}
    allowed = {pending['commit']}
    if pending.get('prior_commit') is not None:
        allowed.add(pending['prior_commit'])
    check((not refs.strip() and pending.get('prior_commit') is None)
          or (refmap.get('HEAD') == refmap.get('refs/heads/main') and refmap.get('HEAD') in allowed),
          'Remote changed beyond the queued formal publication')
    configure_task_git_auth(recorder, clone)
    recorder.command('queued_push', ['git', '-C', clone, 'push', 'origin', 'HEAD:refs/heads/main'], timeout=120)
    expected_remote(recorder, 'queued_remote_verified', pending['commit'])
    names = (STUDENT + '_report.pdf', STUDENT + '.zip')
    verify_raw_public(pending['commit'], {name: {'sha256': pending['files_sha256'][name],
        'bytes': (clone / name).stat().st_size} for name in names}, max_attempts=1, timeout=15)
    check(destination(destination_path) == selected, 'Destination changed during queued publication')
    event = {'scope': pending['scope'], 'commit': pending['commit'], 'tree': pending['tree'],
             'files_sha256': pending['files_sha256'], 'published_at_utc': now(),
             'report_and_zip_public_sha_verified': True}
    if event['scope'] == BASELINE_SCOPE:
        selected['baseline_publication'] = event
    selected.setdefault('publication_history', []).append(event)
    selected.update(publishing_completed=True, publication_scope=event['scope'], commit=event['commit'],
                    publication_tree=event['tree'], publication_files_sha256=event['files_sha256'],
                    published_at_utc=event['published_at_utc'], report_and_zip_public_sha_verified=True)
    atomic_json(destination_path, selected)
    return selected


def flush_publication_queue(recorder, destination_path):
    """Called by the cloud worker while holding the artifact lock; local work never waits for success."""
    path = ROOT / 'analysis/formal_publication_queue.json'
    queue = read(path)
    try:
        if queue.get('pending_commit'):
            resume_formal_commit(recorder, destination_path, queue['pending_commit'])
            queue['pending_commit'] = None
            atomic_json(path, queue)
        ready = queue['latest_ready']
        source = local(ROOT / ready['source'])
        check(sha(source / 'BUNDLE_MANIFEST.json') == ready['source_manifest_sha256'], 'Queued source changed after validation')
        selected = destination(destination_path)
        already_done = (selected.get('publishing_completed') is True
            and selected.get('publication_scope') == ready['scope']
            and selected.get('publication_files_sha256', {}).get(STUDENT + '_report.pdf') == ready['report_sha256'])
        if not already_done:
            for key in ('local_commit', 'publication_tree', 'publication_files_sha256', 'publication_scope', 'prior_publication_commit'):
                recorder.value.pop(key, None)
            publish(SimpleNamespace(baseline_only=ready['scope'] == BASELINE_SCOPE, destination=destination_path), recorder, source)
            selected = destination(destination_path)
        queue.update(status='published_and_verified', updated_at_utc=now(), pending_commit=None,
                     published_commit=selected['commit'], last_error=None)
        atomic_json(path, queue)
        return {'scope': ready['scope'], 'commit': selected['commit'], 'published': True}
    except Exception as error:
        if not queue.get('pending_commit') and recorder.value.get('local_commit'):
            queue['pending_commit'] = {'scope': recorder.value['publication_scope'],
                'commit': recorder.value['local_commit'], 'tree': recorder.value['publication_tree'],
                'files_sha256': recorder.value['publication_files_sha256'],
                'prior_commit': recorder.value.get('prior_publication_commit')}
        queue.update(status='waiting_to_retry', last_error=redact(error), updated_at_utc=now())
        atomic_json(path, queue)
        raise


def finalize(args):
    local(args.state)
    scope = BASELINE_SCOPE if args.baseline_only else FULL_SCOPE
    recorder = Recorder(args)
    try:
        # Waiting must not own the artifact lock: the pipeline invokes the
        # baseline finalizer before its later full-completion event exists.
        if not args.baseline_only:
            recorder.step('wait_for_real_pipeline', lambda: wait_for_pipeline(args, recorder))
        lock_path = local(ROOT / 'analysis/finalizer_artifacts.lock')
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            lock_deadline = time.monotonic() + 900
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as error:
                    check(time.monotonic() < lock_deadline,
                          'Timed out waiting for the progress upload; preserve its work')
                    time.sleep(2)
            selected = destination(args.destination)
            recorder.value['destination_input_sha256'] = sha(args.destination)
            required_inputs = [ROOT / 'results/baseline_evaluation.json']
            if not args.baseline_only:
                required_inputs.append(ROOT / 'results/ablation/ablation_results.json')
            evaluation, _ = recorder.step('validate_real_results', lambda: validate_results(args.baseline_only),
                                          inputs=required_inputs)
            recorder.status('finalizing_local_artifacts', scope=scope)
            recorder.step('update_readme_from_real_results',
                          lambda: update_readme(evaluation, selected['report_url'], args.baseline_only),
                          inputs=(ROOT / 'README.md', ROOT / 'results/baseline_evaluation.json'))
            report_python = ROOT / '.venv_report/bin/python'
            generation_python = ROOT / '.venv/bin/python'
            check(report_python.is_file() and generation_python.is_file(), 'Prepared task-local Python environments are missing')
            report_command = [report_python, ROOT / 'tools/build_report.py', '--artifact-url', selected['report_url']]
            if args.baseline_only:
                report_command.append('--baseline-only')
            recorder.step('build_formal_report', lambda: recorder.command('build_formal_report', report_command),
                inputs=(ROOT / 'tools/build_report.py', args.destination))
            report = ROOT / (STUDENT + '_report.pdf')
            validate_formal_report(report, scope)
            source = local(ROOT / 'deliverables' / (STUDENT + ('_source_BASELINE' if args.baseline_only else '_source')))
            source_command = [generation_python, ROOT / 'tools/package_source.py', '--output-dir', source]
            if args.baseline_only:
                source_command.append('--baseline-only')
            recorder.step('build_formal_source_bundle', lambda: recorder.command('build_formal_source_bundle',
                source_command),
                inputs=(ROOT / 'tools/package_source.py', report, report.with_suffix('.checks.json')))
            bundle = read(source / 'BUNDLE_MANIFEST.json')
            check(bundle.get('status') == 'FORMAL_VALIDATED_LOCAL_BUNDLE' and bundle.get('scope') == scope
                  and bundle.get('validation', {}).get('scope') == scope
                  and bundle['validation'].get('validated') is True,
                  'Formal source bundle is incomplete or belongs to another report scope')
            if args.baseline_only:
                recorder.value['cool_ready'] = recorder.step('preserve_baseline_cool_upload_copy',
                    lambda: freeze_cool_ready(report, source),
                    inputs=(report, report.with_suffix('.checks.json'), ROOT / (STUDENT + '.zip')))
            recorder.value['local_artifact_sha256'] = {str(p.relative_to(ROOT)): sha(local(p, required=True))
                for p in (report, report.with_suffix('.checks.json'), ROOT / (STUDENT + '.zip'),
                          source.with_suffix('.zip'), source / 'BUNDLE_MANIFEST.json')}
            recorder.status('local_formal_artifacts_complete', published=False)
            if args.publish_github:
                result = recorder.step('queue_github_publication', lambda: enqueue_publication(args, source),
                                       inputs=(source / 'BUNDLE_MANIFEST.json', args.destination))
                recorder.status('complete_local_publication_queued', published=False, publication_pending=True,
                                queued_publication=result,
                                completed_at_utc=now())
            else:
                recorder.status('complete_local_only_not_published', published=False, completed_at_utc=now())
            return recorder.value
    except Exception as error:
        recorder.status('stopped_without_claiming_completion', error=redact(error),
                        local_artifacts_preserved=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true', help='List intended steps without running anything')
    parser.add_argument('--publish-github', action='store_true', help='Publish only validated final artifacts to the user-selected public HW3 repository')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Immediately validate/finalize the required 100-clip artifacts without waiting for extras')
    parser.add_argument('--max-wait-hours', type=float, default=18)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--pipeline-state', type=Path, default=ROOT / 'analysis/pipeline_state_v6.json')
    parser.add_argument('--destination', type=Path, default=ROOT / 'analysis/artifact_destination.json')
    parser.add_argument('--state', type=Path,
                        help='Defaults to baseline_finalization_state.json or finalization_state.json by scope')
    args = parser.parse_args()
    if args.state is None:
        args.state = ROOT / 'analysis' / ('baseline_finalization_state.json' if args.baseline_only else 'finalization_state.json')
    try:
        check(math.isfinite(args.max_wait_hours) and 0 < args.max_wait_hours <= 18, 'Wait limit must be positive and at most 18 hours')
        check(math.isfinite(args.poll_seconds) and 1 <= args.poll_seconds <= 60, 'Poll interval must be 1..60 seconds')
        if args.plan:
            steps = ([] if args.baseline_only else ['Wait for ' + READY + ' without holding the artifact lock (maximum 18 hours)'])
            steps += [('Immediately validate real baseline100, mandatory metrics and COOL ZIP; do not read or wait for extras'
                       if args.baseline_only else 'Validate real full-100 evaluation, 4x20 ablation and genuine MP3/COOL ZIP bytes'),
                     'Update README Current status from actual metrics and retain proxy limitations',
                     'Build validated formal report with matching report_scope and recorded artifact URL',
                     'Build fresh validated compact source folder/ZIP with matching scope', 'Record step logs and artifact SHA256']
            if args.baseline_only:
                steps += ['Preserve baseline report/checks/COOL ZIP in immutable deliverables/cool_ready']
            if args.publish_github:
                steps += ['Persist an immutable validated publication queue without requiring network; return to local experiments',
                          'Background sync worker retries cloud publication and interrupted owned commits after reconnection',
                          'Require empty remote for first publication; updates only accept our verified progress/baseline commits',
                          'Verify clean isolated clone HEAD/tree/all file hashes, then copy only the authorized whitelist',
                          'Commit normally on main; preserve prior other files, recheck remote, never force-push',
                          'Verify remote HEAD/main and unauthenticated public report/ZIP bytes']
            print(json.dumps({'plan_only': True, 'repository': REPOSITORY,
                              'scope': BASELINE_SCOPE if args.baseline_only else FULL_SCOPE,
                              'publish_github': args.publish_github, 'steps': steps}, indent=2))
            return
        finalize(args)
    except (ValueError, OSError, KeyError, TypeError, HTTPError, URLError, subprocess.TimeoutExpired) as error:
        parser.exit(1, 'Finalization stopped: ' + redact(error) + '\n')


if __name__ == '__main__':
    main()
