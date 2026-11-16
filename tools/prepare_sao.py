"""Pin and download only the Diffusers files of the authorized course model.

The user must personally accept the model terms and finish local HF login.
This script never accepts terms, displays credentials or downloads the duplicate
stable-audio-tools checkpoint format. Metadata-only is the default.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import certifi
import httpx
from huggingface_hub import HfApi, get_token, get_hf_file_metadata, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from download_ranges import download, sha, write

REPO = 'stabilityai/stable-audio-open-1.0'
REVISION = 'f21265c1e2710b3bd2386596943f0007f55f802e'
DIRECTORIES = {'scheduler', 'text_encoder', 'tokenizer', 'projection_model', 'transformer', 'vae'}


def metadata():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=get_token())
    if info.sha != REVISION: raise ValueError('Pinned model revision changed')
    files = []
    for entry in info.siblings:
        path = Path(entry.rfilename)
        if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe remote path')
        if entry.rfilename not in {'model_index.json', 'README.md', 'LICENSE', 'LICENSE.md'} and path.parts[0] not in DIRECTORIES:
            continue
        if path.suffix.lower() in {'.bin', '.ckpt'}: continue
        lfs = entry.lfs
        files.append({'path': entry.rfilename, 'bytes': entry.size,
                      'lfs_sha256': getattr(lfs, 'sha256', None) if lfs else None,
                      'git_blob_id': entry.blob_id})
    names = {f['path'] for f in files}
    if not {'model_index.json', 'transformer/config.json', 'vae/config.json',
            'projection_model/config.json'}.issubset(names):
        raise ValueError('Required Diffusers component files are absent')
    if not any(f['path'].startswith('transformer/') and f['lfs_sha256'] for f in files):
        raise ValueError('No transformer weight file in the pinned source')
    if any(not isinstance(f['bytes'], int) or f['bytes'] < 0 for f in files):
        raise ValueError('All model file sizes must be available before downloading')
    return {'repository': REPO, 'revision': REVISION,
            'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'files': sorted(files, key=lambda f: f['path']),
            'total_bytes': sum(f['bytes'] or 0 for f in files),
            'duplicate_root_checkpoint_formats_excluded': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', action='store_true')
    p.add_argument('--directory', type=Path, default=ROOT / ('cache/models/stable-audio-open-1.0-' + REVISION[:8]))
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    record = metadata()
    (ROOT / 'analysis').mkdir(exist_ok=True)
    write(ROOT / 'analysis/sao_source_metadata.json', record)
    if not args.download:
        print(json.dumps({'repository': REPO, 'revision': REVISION,
                          'total_bytes': record['total_bytes'], 'files': record['files']}, indent=2)); return
    token = get_token()
    if not token: raise SystemExit('Local HF login is still missing; no model file has been requested')
    # A metadata request to the actual gated weights verifies account rights.
    weight = next(f for f in record['files'] if f['path'].startswith('transformer/') and f['lfs_sha256'])
    get_hf_file_metadata(hf_hub_url(REPO, weight['path'], revision=REVISION), token=token)
    args.directory.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        target = args.directory / entry['path']; target.parent.mkdir(parents=True, exist_ok=True)
        if entry['lfs_sha256']:
            if entry['bytes'] > 8 * 1024 * 1024:
                download(REPO, REVISION, entry['path'], target, entry['bytes'], entry['lfs_sha256'],
                         workers=args.workers, authenticated=True)
                return {**entry, 'actual_sha256': sha(target)}
            if target.is_file() and target.stat().st_size == entry['bytes'] and sha(target) == entry['lfs_sha256']:
                return {**entry, 'actual_sha256': sha(target)}
        elif target.is_file():
            data = target.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob == entry['git_blob_id']: return {**entry, 'actual_sha256': sha(target)}
            raise ValueError('An existing local model config changed; preserve it')
        url = hf_hub_url(REPO, entry['path'], revision=REVISION)
        with httpx.Client(follow_redirects=True, verify=certifi.where(), timeout=120) as client:
            response = client.get(url, headers={'Authorization': 'Bearer ' + token})
            if response.status_code != 200:
                raise RuntimeError(f"Pinned model file HTTP {response.status_code}; no credential or redirect URL is logged")
            data = response.content
        if len(data) != entry['bytes']: raise ValueError('Model/config file size differs from the pinned source')
        if entry['lfs_sha256']:
            if hashlib.sha256(data).hexdigest() != entry['lfs_sha256']: raise ValueError('Model file hash differs')
        elif hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != entry['git_blob_id']:
            raise ValueError('Model config Git blob hash differs')
        tmp = target.with_name(target.name + '.tmp'); tmp.write_bytes(data); tmp.replace(target)
        return {**entry, 'actual_sha256': sha(target)}

    completed = []
    small = [f for f in record['files'] if f['bytes'] <= 8 * 1024 * 1024]
    large = [f for f in record['files'] if f['bytes'] > 8 * 1024 * 1024]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch, f) for f in small]): completed.append(future.result())
    # Each large file already uses bounded parallel byte-range workers.
    for entry in large:
        print(json.dumps({'downloading_component': entry['path'], 'bytes': entry['bytes']}), flush=True)
        completed.append(fetch(entry))
        write(ROOT / 'analysis/sao_download_progress.json', {'completed_files': completed,
                'total_files': len(record['files']), 'complete': False})
    result = {**record, 'completed': True, 'files': sorted(completed, key=lambda f: f['path']),
              'local_directory': str(args.directory.resolve().relative_to(ROOT)),
              'downloaded_at_utc': datetime.now(timezone.utc).isoformat(), 'terms_accepted_by_agent': False}
    write(args.directory / 'model_provenance.json', result)
    write(ROOT / 'analysis/sao_download_progress.json', result)
    print(json.dumps({'complete': True, 'model_path': result['local_directory'], 'revision': REVISION}), flush=True)


if __name__ == '__main__': main()
