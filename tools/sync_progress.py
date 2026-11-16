"""Publish checked HW3 progress snapshots while local generation continues.

Uses the existing Git credential helper and only the user's selected repository.
Shares the finalizer artifact lock; yields permanently to a formal publication.
Progress is never marked as a validated submission. No weights/captions/cache.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import generate
from tools import finalize_when_ready as f

STATE = ROOT / 'analysis/progress_sync_state.json'
DESTINATION = ROOT / 'analysis/artifact_destination.json'


def prepare(recorder):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    draft = ROOT / 'deliverables/progress_snapshots' / (stamp + '_DRAFT')
    recorder.command('prepare_snapshot', [ROOT / '.venv/bin/python', '-B',
        ROOT / 'tools/package_source.py', '--baseline-only', '--draft', '--output-dir', draft])
    inventory = f.read(draft / 'BUNDLE_MANIFEST.json')
    f.check(inventory.get('status') == 'DRAFT_INCOMPLETE_NOT_FOR_SUBMISSION', 'Not a progress snapshot')
    manifest = f.read(draft / 'results/baseline/manifest.json')
    f.check(manifest['run_sha256'] == generate.text_sha(generate.canonical(manifest['identity'])),
            'Generation identity differs from its recorded run hash')
    f.check(manifest['identity']['protocol']['sha256'] == f.sha(ROOT / 'analysis/experiment_protocol_frozen.json'),
            'Progress snapshot uses a different frozen protocol')
    f.check(all(f.sha(ROOT / n) == h for n, h in manifest['identity']['source_sha256'].items()),
            'Generation source changed before progress publication')
    captions = {row['id']: row['caption'] for row in generate.read_captions(ROOT / 'sample_code/test.csv')}
    records = {row['id']: row for row in manifest.get('samples', [])}
    f.check(len(records) == len(manifest.get('samples', [])), 'Duplicate progress audio IDs')
    missing_masters = []
    for identifier, row in records.items():
        caption_sha = generate.text_sha(captions[identifier])
        expected = generate.text_sha(generate.canonical({'run_sha256': manifest['run_sha256'],
            'id': identifier, 'caption_sha256': caption_sha}))
        f.check(row['contract_sha256'] == expected, 'Progress record contract changed')
        mp3 = f.local(ROOT / 'results/baseline' / row['mp3']['path'], required=True)
        f.check(f.sha(mp3) == row['mp3']['sha256'], 'Recorded progress MP3 changed')
        master = f.local(ROOT / 'results/baseline' / row['wav']['path'])
        if not master.is_file():
            missing_masters.append(identifier)
        f.check(row['caption_sha256'] == caption_sha and row['seed'] == 20261115 + int(identifier),
                'Progress caption or noise seed changed')
    recorder.value['missing_local_wav_masters'] = missing_masters
    recorder.save()
    public = ROOT / 'deliverables/progress_snapshots' / (stamp + '_public')
    public.mkdir(parents=True, exist_ok=False)
    for name, record in inventory['files'].items():
        if name in {f.STUDENT + '_report.pdf', f.STUDENT + '_report.checks.json'}:
            continue  # Formal deliverables are published only by the finalizer.
        source = f.local(draft / name, required=True)
        f.public_member(name, source)
        f.check(f.sha(source) == record['sha256'], 'Prepared progress file hash changed')
        if name.startswith('results/baseline/generated_samples/'):
            # A just-finished codec file can appear before its record is committed.
            # Only publish audio present in the copied generation manifest.
            identifier = Path(name).stem
            if identifier not in records:
                continue
            f.check(record['sha256'] == records[identifier]['mp3']['sha256'], 'Progress MP3 hash differs')
        target = f.local(public / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    f.check({p.stem for p in (public / 'results/baseline/generated_samples').glob('*.mp3')} == set(records),
            'Progress snapshot is missing recorded MP3 files')
    pipeline = f.read(ROOT / 'analysis/pipeline_state_v6.json')
    snapshot = {'status': 'WORK_IN_PROGRESS_NOT_FOR_SUBMISSION', 'snapshot_only': True,
                'updated_at_utc': f.now(), 'generated_clips': len(records), 'required_clips': 100,
                'pipeline_stage_at_snapshot': pipeline.get('current_stage'),
                'required_evaluation_complete': pipeline.get('stages', {}).get('required_metrics', {}).get('completed') is True,
                'formal_submission_validated': False, 'original_dataset_included': False,
                'completed_audio_ids': sorted(records, key=int)}
    f.atomic_json(public / 'PROGRESS_SNAPSHOT.json', snapshot)
    (public / 'README.md').write_text(
        '# HW3：製作進度（尚非正式繳交版）\n\n'
        f"學號：r14942154。更新：{snapshot['updated_at_utc']}。\n\n"
        f"已完成 **{len(records)} / 100 首** baseline 音樂；100 題 CLAP 檢索已完成。\n\n"
        '此處先公開程式、模型固定來源、檢索結果與已完成的 MP3。正式報告與 COOL ZIP '
        '會在全部必要生成、評估與格式驗證後更新。這份進度快照不能作為完整作業繳交。\n\n'
        '`PROGRESS_SNAPSHOT.json` 是該次上傳的歷史快照。生成在本機持續進行，'
        '每新增約 5 首或有新的執行階段時分批更新。\n\n'
        '音檔：`results/baseline/generated_samples/`；檢索：`results/retrieval/`；'
        '固定設定：`analysis/experiment_protocol_frozen.json`。\n\n'
        '環境使用 Python 3.11，版本見 requirements 與 lock 檔。原始課程 caption CSV、'
        'MusicCaps 資料集、模型權重與登入快取均未公開。重現時須自行備妥原始課程資料，'
        '並先親自接受官方模型條款；使用 `reproduce.py` 與原始固定設定重現時，'
        '需等完整 100 首的正式 manifest 完成。\n\n'
        'SAO/CLAP 依固定 HF LFS 完整雜湊驗證；VGGish/PaSST 使用固定鏡像 SHA 與'
        '官方 release 大小核對。模型與程式授權資訊見各來源及 `licenses/`。\n', encoding='utf-8')
    return public, snapshot


def publish(recorder, public, snapshot):
    selected = f.destination(DESTINATION)
    if selected.get('publishing_completed') is True:
        return {'formal_publication_complete': True}
    prior = None
    if selected.get('commit'):
        prior = f.previous_baseline_publication(selected, f.BASELINE_SCOPE)
        f.check(prior.get('scope') == f.PROGRESS_SCOPE, 'Formal ownership cannot be replaced by progress')
    f.public_repo(recorder, 'progress_repo_public')
    if prior is None:
        f.empty_remote(recorder, 'progress_empty_remote')
    else:
        f.expected_remote(recorder, 'progress_expected_remote', prior['commit'])
    clone = f.local(ROOT / 'github_progress')
    if not clone.exists():
        recorder.command('progress_clone', ['git', '-c', 'protocol.file.allow=never', 'clone', '--', f.REPOSITORY, clone])
    f.configure_task_git_auth(recorder, clone)
    for flag in ([], ['--push']):
        origin = recorder.command('progress_origin' + ('_push' if flag else ''),
            ['git', '-C', clone, 'remote', 'get-url', *flag, 'origin']).strip()
        f.check(origin == f.REPOSITORY, 'Progress clone has a different destination')
    prior_files = {} if prior is None else prior['files_sha256']
    if prior:
        f.verify_owned_tree(recorder, clone, prior['commit'], prior_files, prior['tree'], 'progress_owned')
    else:
        f.check({p.name for p in clone.iterdir()} == {'.git'}, 'First progress clone contains unexpected work')
        recorder.command('progress_main', ['git', '-C', clone, 'symbolic-ref', 'HEAD', 'refs/heads/main'])
    copied = {}
    for source in sorted(public.rglob('*')):
        if not source.is_file():
            continue
        name = str(source.relative_to(public)); f.public_member(name, source)
        target = f.local(clone / name)
        if target.exists():
            f.check(name in prior_files and f.sha(target) == prior_files[name], 'Progress target has external changes')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        f.check(f.sha(target) == f.sha(source), 'Progress copy mismatch')
        copied[name] = f.sha(target)
    changed = {name for name, digest in copied.items() if prior_files.get(name) != digest}
    f.check(changed, 'No changed progress files')
    for key, fallback in [('user.name', f.STUDENT), ('user.email', 'alexsu30812-gif@users.noreply.github.com')]:
        existing = recorder.command('progress_get_' + key, ['git', '-C', clone, 'config', '--get', key], okay=(0, 1)).strip()
        if not existing:
            recorder.command('progress_set_' + key, ['git', '-C', clone, 'config', '--local', key, fallback])
    recorder.command('progress_add', ['git', '-C', clone, 'add', '--', *sorted(copied)])
    staged = recorder.command('progress_staged', ['git', '-C', clone, 'diff', '--cached', '--name-only', '-z'])
    f.check(set(filter(None, staged.split('\0'))) == changed, 'Unexpected progress index entries')
    recorder.command('progress_commit', ['git', '-C', clone, 'commit', '-m',
        f"Sync HW3 progress: {snapshot['generated_clips']}/100 baseline clips (not final submission)"])
    commit = recorder.command('progress_commit_id', ['git', '-C', clone, 'rev-parse', 'HEAD']).strip()
    files = {**prior_files, **copied}
    tree = f.verify_owned_tree(recorder, clone, commit, files, label='progress_committed')
    if prior:
        parent = recorder.command('progress_parent', ['git', '-C', clone, 'rev-parse', 'HEAD^']).strip()
        f.check(parent == prior['commit'], 'Progress commit does not extend its recorded parent')
        f.expected_remote(recorder, 'progress_remote_before_push', prior['commit'])
    else:
        f.empty_remote(recorder, 'progress_remote_before_push')
    event = {'scope': f.PROGRESS_SCOPE, 'commit': commit, 'tree': tree, 'files_sha256': files,
             'generated_clips': snapshot['generated_clips'], 'published_at_utc': f.now(),
             'pipeline_stage_at_snapshot': snapshot['pipeline_stage_at_snapshot'],
             'remote_and_public_snapshot_verified': False}
    return push_and_verify(recorder, selected, event, clone, snapshot)


def push_and_verify(recorder, selected, event, clone, snapshot):
    commit, files, tree = event['commit'], event['files_sha256'], event['tree']
    recorder.value['pending_publication'] = event
    recorder.status('pushing_progress', uploading_clips=snapshot['generated_clips'])
    recorder.command('progress_push', ['git', '-C', clone, 'push', 'origin', 'HEAD:refs/heads/main'], timeout=600)
    f.expected_remote(recorder, 'progress_remote_after_push', commit)
    verify_names = ['README.md', 'PROGRESS_SNAPSHOT.json']
    if snapshot['completed_audio_ids']:
        verify_names.append('results/baseline/generated_samples/' + snapshot['completed_audio_ids'][-1] + '.mp3')
    f.verify_raw_public(commit, {name: {'sha256': files[name], 'bytes': (clone / name).stat().st_size}
                               for name in verify_names}, max_attempts=1, timeout=15)
    f.check(f.destination(DESTINATION) == selected, 'Destination changed during progress publication')
    event['remote_and_public_snapshot_verified'] = True
    selected.update(publication_scope=f.PROGRESS_SCOPE, commit=commit, publication_tree=tree,
        publication_files_sha256=files, progress_publication=event, publishing_completed=False)
    selected.setdefault('publication_history', []).append(event)
    f.atomic_json(DESTINATION, selected)
    recorder.status('progress_publication_verified', pending_publication=None,
                    uploaded_clips=snapshot['generated_clips'], commit=commit)
    return event


def resume_pending(recorder, pending):
    """Retry only a previously checked, still-clean owned commit after auth/network failure."""
    selected = f.destination(DESTINATION)
    f.check(pending.get('scope') == f.PROGRESS_SCOPE, 'Pending publication has an unexpected scope')
    clone = f.local(ROOT / 'github_progress')
    f.verify_owned_tree(recorder, clone, pending['commit'], pending['files_sha256'], pending['tree'], 'pending_progress')
    origin = recorder.command('pending_origin', ['git', '-C', clone, 'remote', 'get-url', 'origin']).strip()
    push_origin = recorder.command('pending_push_origin', ['git', '-C', clone, 'remote', 'get-url', '--push', 'origin']).strip()
    f.check(origin == f.REPOSITORY and push_origin == f.REPOSITORY, 'Pending clone has a different destination')
    refs = recorder.command('pending_remote', ['git', 'ls-remote', f.REPOSITORY], timeout=120)
    refmap = {line.split()[1]: line.split()[0] for line in refs.splitlines() if line.strip()}
    expected = selected.get('commit')
    allowed = {pending['commit']}
    if expected is not None:
        allowed.add(expected)
    f.check((not refs.strip() and expected is None)
            or (refmap.get('HEAD') == refmap.get('refs/heads/main')
                and refmap.get('HEAD') in allowed),
            'Remote changed beyond the pending owned publication')
    f.configure_task_git_auth(recorder, clone)
    snapshot = f.read(clone / 'PROGRESS_SNAPSHOT.json')
    f.check(snapshot.get('status') == 'WORK_IN_PROGRESS_NOT_FOR_SUBMISSION'
            and snapshot.get('generated_clips') == pending['generated_clips'], 'Pending snapshot differs')
    return push_and_verify(recorder, selected, pending, clone, snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'analysis/progress_sync.lock').open('a') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            selected = f.destination(DESTINATION)
            queue_path = ROOT / 'analysis/formal_publication_queue.json'
            queue = f.read(queue_path) if queue_path.is_file() else {}
            if (selected.get('publication_scope') == f.FULL_SCOPE and selected.get('publishing_completed') is True
                    and (not queue or queue.get('status') == 'published_and_verified')):
                print('Full formal publication verified; cloud worker finished.', flush=True)
                return
            previous = selected.get('progress_publication', {})
            manifest = f.read(ROOT / 'results/baseline/manifest.json')
            pipeline = f.read(ROOT / 'analysis/pipeline_state_v6.json')
            new_count = len(manifest.get('samples', []))
            previous_count = previous.get('generated_clips', -5)
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(previous['published_at_utc'])).total_seconds() if previous else 601
            changed_stage = previous.get('pipeline_stage_at_snapshot') != pipeline.get('current_stage')
            saved = f.read(STATE) if STATE.is_file() else {}
            pending_progress = bool(saved.get('pending_publication'))
            formal_pending = bool(queue and queue.get('status') != 'published_and_verified')
            progress_due = (not queue and selected.get('publishing_completed') is not True
                and (not previous or new_count - previous_count >= 5 or changed_stage
                     or (new_count > previous_count and elapsed >= 600)))
            due = pending_progress or formal_pending or progress_due
            if due:
                with (ROOT / 'analysis/finalizer_artifacts.lock').open('a') as shared:
                    try:
                        fcntl.flock(shared, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        if args.once:
                            raise SystemExit('Formal finalizer is active; no duplicate publication was started')
                    else:
                        saved = f.read(STATE) if STATE.is_file() else {}
                        recorder = f.Recorder(SimpleNamespace(state=STATE, baseline_only=True, publish_github=True))
                        recorder.value['scope'] = f.PROGRESS_SCOPE
                        recorder.value['background_cloud_retry'] = True
                        try:
                            if saved.get('pending_publication'):
                                resume_pending(recorder, saved['pending_publication'])
                            elif queue_path.is_file():
                                current_queue = f.read(queue_path)
                                if current_queue.get('status') != 'published_and_verified':
                                    result = f.flush_publication_queue(recorder, DESTINATION)
                                    recorder.status('formal_publication_verified', **result)
                            else:
                                public, snapshot = prepare(recorder)
                                publish(recorder, public, snapshot)
                        except Exception as error:
                            recorder.status('cloud_waiting_to_retry', error=f.redact(error))
                            if args.once:
                                raise
            if args.once:
                return
            time.sleep(60)


if __name__ == '__main__':
    main()
