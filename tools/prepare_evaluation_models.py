"""Verify/download the fixed VGGish and PaSST evaluator checkpoints.

Without --download this prints source metadata and changes nothing. Downloads
use bounded resumable HF byte ranges and verify the full pinned mirror LFS
SHA256 before publication. Original GitHub releases expose sizes, but their
release-asset digest is null: the mirror SHA is not claimed as an independently
verified full official-release digest. VGGish's official filename additionally
identifies the matching eight-character SHA prefix.

Bad existing targets are preserved in cache/quarantine/evaluator_weights with
their actual SHA and a JSON explanation; they are never silently overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from download_ranges import download, sha, write

SOURCES = {
    'vggish': {
        'target': 'cache/models/vggish-10086976.pth',
        'original_release': {
            'repository': 'harritaylor/torchvggish', 'release_tag': 'v0.1', 'asset_id': 15146902,
            'filename': 'vggish-10086976.pth', 'bytes': 288567937, 'digest': None,
            'url': 'https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth',
            'filename_sha256_prefix': '10086976'},
        'transport': {
            'type': 'huggingface_mirror', 'repository': 'YannQi/COMBO-AVS-checkpoints',
            'revision': '0cccd93230490a170b26f6d3bc83a88b88a11ac1',
            'path': 'pretrained/vggish-10086976.pth', 'bytes': 288567937,
            'sha256': '10086976245803799d9194e9a73d9b6c1549c71d1b80106f5cade5608a561f4b'}},
    'passt': {
        'target': 'cache/torch/hub/checkpoints/passt-s-f128-p16-s10-ap.476-swa.pt',
        'original_release': {
            'repository': 'kkoutini/PaSST', 'release_tag': 'v0.0.1-audioset', 'asset_id': 47268195,
            'filename': 'passt-s-f128-p16-s10-ap.476-swa.pt', 'bytes': 344665734, 'digest': None,
            'url': 'https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt',
            'filename_sha256_prefix': None},
        'transport': {
            'type': 'huggingface_mirror', 'repository': 'OpenMOSS-Team/OmniVAE',
            'revision': '6bb9ab6dedad19051600c832c441d6cb53661aa0',
            'path': 'eval/models/t2av/torch_cache/hub/checkpoints/passt-s-f128-p16-s10-ap.476-swa.pt',
            'bytes': 344665734,
            'sha256': '302903fa8c4aee817b11dc982da0b29aaf8d11a3e722420476d0a12c9db70c2c'}}}


def checked_source(kind):
    if kind not in SOURCES: raise ValueError('Choose vggish or passt explicitly')
    source = SOURCES[kind]
    official, mirror = source['original_release'], source['transport']
    if official['bytes'] != mirror['bytes']:
        raise ValueError('Pinned official-release and mirror sizes disagree')
    prefix = official['filename_sha256_prefix']
    if prefix and not mirror['sha256'].startswith(prefix):
        raise ValueError('VGGish mirror SHA does not match the official filename hash prefix')
    return source


def verification(path, source):
    path = Path(path)
    if path.is_symlink(): raise ValueError('Evaluator checkpoint targets must not be symlinks')
    if not path.is_file(): raise ValueError('Evaluator checkpoint is missing or not a regular file')
    size, actual = path.stat().st_size, sha(path)
    expected = source['transport']
    return {'bytes': size, 'sha256': actual,
            'valid': size == source['original_release']['bytes'] == expected['bytes']
                     and actual == expected['sha256']}


def quarantine(path, actual, source, root=ROOT):
    """Move one bad regular file to a new SHA-named event directory."""
    path, root = Path(path), Path(root)
    directory = root / 'cache/quarantine/evaluator_weights'
    directory.mkdir(parents=True, exist_ok=True)
    event = directory / (path.name + '.' + actual['sha256'] + '.' + uuid.uuid4().hex[:8])
    event.mkdir()  # Exclusive ownership; never replaces an earlier quarantine.
    destination = event / path.name
    note = {'reason': 'Existing evaluator checkpoint size/full SHA differs from the pinned source; preserved before retry.',
            'original_path': str(path.resolve()), 'quarantined_path': str(destination.resolve()),
            'actual_bytes': actual['bytes'], 'actual_sha256': actual['sha256'],
            'expected_official_release_bytes': source['original_release']['bytes'],
            'expected_mirror_bytes': source['transport']['bytes'],
            'expected_mirror_sha256': source['transport']['sha256'],
            'official_release_full_digest': source['original_release']['digest'],
            'quarantined_at_utc': datetime.now(timezone.utc).isoformat()}
    # Write the preservation reason before the move; if the move fails, the
    # original file remains and the event still explains the attempted action.
    write(event / 'quarantine.json', note)
    path.rename(destination)
    print(json.dumps({'quarantined_checkpoint': path.name, 'actual_bytes': actual['bytes'],
                      'actual_sha256': actual['sha256'], 'preserved_at': str(destination)}), flush=True)
    return destination


def ensure(kind, *, download_missing=False, workers=16, root=ROOT):
    """Return (fully verified local path, stable transport provenance).

    The CLI controls downloading explicitly. Encoder loading may request the
    same bounded preparation, replacing the previous fragile Torch Hub fetch.
    """
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError('Use 1..16 download workers')
    source = checked_source(kind); root = Path(root)
    target = root / source['target']; target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink(): raise ValueError('Evaluator checkpoint target must not be a symlink')
    import fcntl
    with target.with_name(target.name + '.prepare.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another preparation owns this checkpoint; wait for it to finish') from exc
        if target.exists():
            actual = verification(target, source)
            if not actual['valid']:
                quarantine(target, actual, source, root)
        if not target.exists():
            if not download_missing:
                raise ValueError('Prepare this checkpoint with --download --kind ' + kind + ' before loading it')
            mirror = source['transport']
            download(mirror['repository'], mirror['revision'], mirror['path'], target,
                     mirror['bytes'], mirror['sha256'], workers=workers, authenticated=False)
        actual = verification(target, source)
        if not actual['valid']:
            quarantine(target, actual, source, root)
            raise ValueError('Evaluator download failed full-size/full-SHA verification; loading is forbidden')
        provenance = {'original_release': source['original_release'], 'transport': source['transport'],
            'verification': {'actual_bytes': actual['bytes'], 'actual_sha256': actual['sha256'],
                'matches_official_release_size': True, 'matches_pinned_mirror_full_sha256': True,
                'official_full_release_digest_available': False,
                'official_full_release_digest_independently_verified': False,
                'official_filename_sha256_prefix': source['original_release']['filename_sha256_prefix'],
                'claim_scope': 'Pinned mirror bytes and original release size; no independent full official digest is available.'},
            'helper_source_sha256': sha(Path(__file__)),
            'range_downloader_source_sha256': sha(ROOT / 'tools/download_ranges.py')}
        return target, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--kind', choices=('vggish', 'passt', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16: parser.error('--workers must be 1..16')
    kinds = ('vggish', 'passt') if args.kind == 'all' else (args.kind,)
    if not args.download:
        print(json.dumps({kind: checked_source(kind) for kind in kinds}, indent=2)); return
    try:
        for kind in kinds:
            path, provenance = ensure(kind, download_missing=True, workers=args.workers)
            print(json.dumps({'kind': kind, 'complete_full_sha256_verified': True,
                              'path': str(path), 'provenance': provenance}, indent=2), flush=True)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, 'Evaluator model preparation stopped: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
