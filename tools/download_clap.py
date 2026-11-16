"""Download the public, SHA-pinned HW3 CLAP checkpoint without HF credentials.

Metadata-only is the default. --download writes only the requested checkpoint
under the supplied cache directory; no package cache or credential is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieve import CLAP_BYTES, CLAP_FILE, CLAP_REPO, CLAP_REVISION, CLAP_SHA256, atomic_json, sha256


def public_metadata():
    url = f'https://huggingface.co/api/models/{CLAP_REPO}/revision/{CLAP_REVISION}?blobs=true'
    process = subprocess.run(['/usr/bin/curl', '-q', '--fail', '--silent', '--show-error',
                              '--proto', '=https', '--tlsv1.2', '--connect-timeout', '15',
                              '--max-time', '90', url], capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError('Official public metadata request failed; no TLS bypass or credential retry is used')
    data = json.loads(process.stdout)
    if data.get('id') != CLAP_REPO or data.get('sha') != CLAP_REVISION:
        raise ValueError('Official repository revision differs from the pinned source')
    entry = next(row for row in data['siblings'] if row['rfilename'] == CLAP_FILE)
    lfs = entry.get('lfs', {})
    if entry.get('size') != CLAP_BYTES or lfs.get('sha256') != CLAP_SHA256:
        raise ValueError('Official checkpoint metadata changed; preserve existing files and review the source')
    return {'repository': CLAP_REPO, 'revision': CLAP_REVISION, 'filename': CLAP_FILE,
            'expected_bytes': CLAP_BYTES, 'expected_sha256': CLAP_SHA256,
            'architecture': 'HTSAT-base', 'enable_fusion': False,
            'source_url': f'https://huggingface.co/{CLAP_REPO}/resolve/{CLAP_REVISION}/{CLAP_FILE}',
            'metadata_url': url, 'authentication': 'anonymous public download; no HF token sent'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'cache/models')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--transport', choices=['hub', 'curl'], default='hub')
    args = parser.parse_args()
    metadata = public_metadata()
    if not args.download:
        print(json.dumps(metadata, indent=2))
        return
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / CLAP_FILE
    partial = target.with_name(target.name + '.partial')
    partial_meta = target.with_name(target.name + '.partial.json')
    final_meta = target.with_suffix('.provenance.json')
    if target.exists():
        if target.stat().st_size != CLAP_BYTES or sha256(target) != CLAP_SHA256:
            raise ValueError('Existing checkpoint differs; it has been preserved')
    else:
        if partial.exists():
            if not partial_meta.exists() or json.loads(partial_meta.read_text()) != metadata:
                raise ValueError('Unowned or changed partial download; preserve it before retrying')
        else:
            atomic_json(partial_meta, metadata)
        if partial.exists() and partial.stat().st_size > CLAP_BYTES:
            raise ValueError('Partial download is larger than the official file')
        if args.transport == 'hub':
            # The official Xet/Hub client can fetch independent content chunks.
            # It never receives an HF user token for this public repository.
            os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
            os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
            os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
            os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import disable_progress_bars
            disable_progress_bars()
            downloaded = Path(hf_hub_download(repo_id=CLAP_REPO,
                filename=CLAP_FILE, revision=CLAP_REVISION, token=False,
                local_dir=args.directory))
            if downloaded.resolve() != target.resolve():
                raise ValueError('Hub returned a different target path')
            if target.stat().st_size != CLAP_BYTES or sha256(target) != CLAP_SHA256:
                raise ValueError('Hub download fails the pinned official size/SHA256')
        elif not partial.exists() or partial.stat().st_size != CLAP_BYTES:
            # curl -C resumes only a matching partial file; curlrc and auth are disabled.
            command = ['/usr/bin/curl', '-q', '--fail', '--show-error', '--location',
                       '--proto', '=https', '--proto-redir', '=https', '--tlsv1.2',
                       '--connect-timeout', '20', '--retry', '3', '--retry-delay', '2',
                       '--continue-at', '-', '--output', str(partial), metadata['source_url']]
            result = subprocess.run(command)
            if result.returncode:
                raise RuntimeError('Public checkpoint download incomplete; verified partial metadata is retained for resume')
        if args.transport == 'curl':
            if partial.stat().st_size != CLAP_BYTES or sha256(partial) != CLAP_SHA256:
                raise ValueError('Downloaded bytes fail the official size/SHA256; partial file is preserved')
            partial.replace(target)
    metadata.update(actual_bytes=target.stat().st_size, actual_sha256=sha256(target),
                    sha256_matches_official=True, cache_file=str(target.relative_to(ROOT))
                    if target.is_relative_to(ROOT) else target.name)
    atomic_json(final_meta, metadata)
    if partial_meta.exists():
        partial_meta.unlink()
    print(json.dumps({'checkpoint': str(target), 'sha256': metadata['actual_sha256'],
                      'bytes': metadata['actual_bytes'], 'sha256_matches_official': True}, indent=2))


if __name__ == '__main__':
    main()
