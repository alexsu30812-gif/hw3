"""Download an authorized HF file in checked, resumable byte ranges.

Authentication stays in memory and the HF credential store. Neither secrets
nor signed redirect URLs are printed or persisted. Only a complete file with
the pinned SHA256 becomes the requested destination.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import time

import certifi
import httpx


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def download(repo, revision, filename, destination, expected_bytes, expected_sha,
             *, workers=8, chunk_bytes=8 * 1024 * 1024, authenticated=False, prefix=None):
    if not re.fullmatch(r'[a-f0-9]{40}', revision) or not re.fullmatch(r'[a-f0-9]{64}', expected_sha):
        raise ValueError('Pin the actual repository commit and full official SHA256')
    if expected_bytes < 1 or not 1 <= workers <= 16:
        raise ValueError('Invalid file size or concurrency')
    destination = Path(destination).resolve(); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != expected_bytes or sha(destination) != expected_sha:
            raise ValueError('Existing destination differs; preserve it')
        return destination
    url = f'https://huggingface.co/{repo}/resolve/{revision}/{filename}?download=true'
    owned = destination.with_name(destination.name + '.ranges'); owned.mkdir(exist_ok=True)
    metadata = {'repo': repo, 'revision': revision, 'filename': filename,
                'bytes': expected_bytes, 'sha256': expected_sha, 'chunk_bytes': chunk_bytes,
                'authenticated': authenticated}
    manifest = owned / 'source.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise ValueError('Existing chunk directory belongs to different source metadata')
    elif any(owned.iterdir()):
        raise ValueError('Chunk directory has unowned files; preserve them')
    else:
        write(manifest, metadata)
    # The optional caller-owned prefix is untrusted until the final file hash.
    # Complete chunks can be reused, while an append-only downloader is stopped.
    count = (expected_bytes + chunk_bytes - 1) // chunk_bytes
    if prefix and Path(prefix).is_file():
        with Path(prefix).open('rb') as f:
            prefix_bytes = Path(prefix).stat().st_size
            for i in range(min(count, prefix_bytes // chunk_bytes)):
                target = owned / f'{i:06d}.part'
                block = f.read(chunk_bytes)
                if not target.exists() and len(block) == chunk_bytes:
                    tmp = target.with_suffix('.tmp'); tmp.write_bytes(block); tmp.replace(target)
                    write(target.with_suffix('.json'), {'bytes': len(block), 'sha256': sha(target), 'reused_prefix': True})
    token = None
    if authenticated:
        from huggingface_hub import get_token
        token = get_token()
        if not token:
            raise ValueError('Finish local Hugging Face login before downloading gated weights')
    began = time.monotonic()
    completed_bytes = 0

    def fetch(i):
        start = i * chunk_bytes; end = min(expected_bytes, start + chunk_bytes) - 1
        target = owned / f'{i:06d}.part'; note = target.with_suffix('.json')
        if target.exists() and note.exists():
            prior = json.loads(note.read_text())
            if target.stat().st_size == end - start + 1 and prior['sha256'] == sha(target):
                return target.stat().st_size
            raise ValueError(f'Recorded chunk {i} changed; preserve it')
        headers = {'Range': f'bytes={start}-{end}'}
        if token: headers['Authorization'] = 'Bearer ' + token
        last = None
        for attempt in range(4):
            temporary = target.with_suffix('.tmp')
            try:
                with httpx.Client(follow_redirects=True, verify=certifi.where(),
                                  timeout=httpx.Timeout(180.0, connect=30.0), trust_env=True) as client:
                    with client.stream('GET', url, headers=headers) as response:
                        if response.status_code in (401, 403):
                            raise PermissionError('The current HF account does not have authorized file access')
                        if response.status_code != 206 or response.headers.get('content-range') != f'bytes {start}-{end}/{expected_bytes}':
                            raise ValueError('Server did not return the exact requested range')
                        n = 0
                        with temporary.open('wb') as f:
                            for block in response.iter_bytes():
                                n += len(block)
                                if n > end - start + 1: raise ValueError('Oversized range response')
                                f.write(block)
                        if n != end - start + 1: raise ValueError('Incomplete range response')
                temporary.replace(target)
                write(note, {'bytes': n, 'sha256': sha(target), 'reused_prefix': False})
                return n
            except PermissionError:
                raise
            except Exception as error:
                last = type(error).__name__
                if attempt < 3: time.sleep(1 + attempt)
        raise RuntimeError(f'Chunk {i} failed after bounded retries ({last}); no signed URL or credential is logged')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, i) for i in range(count)]
        for n, future in enumerate(as_completed(futures), 1):
            completed_bytes += future.result()
            if n % max(1, count // 40) == 0 or n == count:
                elapsed = time.monotonic() - began
                progress = {'chunks_complete': n, 'chunks_total': count,
                            'bytes_present': completed_bytes, 'total_bytes': expected_bytes,
                            'seconds_this_run': round(elapsed, 2)}
                write(owned / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
    assembled = destination.with_name(destination.name + '.assembling')
    with assembled.open('wb') as out:
        for i in range(count):
            with (owned / f'{i:06d}.part').open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''): out.write(block)
    if assembled.stat().st_size != expected_bytes or sha(assembled) != expected_sha:
        raise ValueError('Full source hash verification failed; chunks and assembled file are preserved')
    assembled.replace(destination)
    write(destination.with_name(destination.name + '.verified.json'),
          {**metadata, 'complete_sha256_verified': True, 'seconds': time.monotonic() - began})
    print(json.dumps({'complete': True, 'file': destination.name, 'bytes': expected_bytes,
                      'sha256': expected_sha}), flush=True)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True); p.add_argument('--revision', required=True)
    p.add_argument('--filename', required=True); p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--bytes', type=int, required=True); p.add_argument('--sha256', required=True)
    p.add_argument('--workers', type=int, default=8); p.add_argument('--chunk-mib', type=int, default=8)
    p.add_argument('--authenticated', action='store_true'); p.add_argument('--reuse-prefix', type=Path)
    a = p.parse_args()
    download(a.repo, a.revision, a.filename, a.destination, a.bytes, a.sha256,
             workers=a.workers, chunk_bytes=a.chunk_mib * 1024 * 1024,
             authenticated=a.authenticated, prefix=a.reuse_prefix)


if __name__ == '__main__': main()
