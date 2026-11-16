"""Build the course ZIP only from complete, verified, genuine output artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import zipfile

from generate import canonical, digest, inspect_mp3, read_captions, text_sha, verify_record
from retrieve import CLAP_SHA256, reference_inventory, validate_prediction

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def archive_check(path, student_id, hashes):
    expected = {f'{student_id}/{name}' for name in hashes}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != expected or z.testzip():
            raise ValueError('ZIP contains missing, duplicate, unexpected or corrupt members')
        for name in names:
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts or any(p.startswith('.') for p in rel.parts):
                raise ValueError('Unsafe ZIP member')
            if (z.getinfo(name).external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('ZIP symlinks are not allowed')
            short = str(Path(*rel.parts[1:]))
            if hashlib.sha256(z.read(name)).hexdigest() != hashes[short]:
                raise ValueError('ZIP bytes differ from the reviewed source file')
    return {'files': len(expected), 'full_zip_readback': True, 'all_file_hashes_match': True}


def package(args):
    if not re.fullmatch(r'[A-Za-z0-9]+', args.student_id): raise ValueError('Invalid student ID')
    rows = read_captions(args.captions)
    manifest = read(args.generation_dir / 'manifest.json')
    identity = manifest['identity']; samples = manifest['samples']
    if not manifest.get('formal_complete') or not manifest.get('complete') or len(samples) != 100:
        raise ValueError('All 100 genuine generated pieces must be complete')
    if identity['mode'] != 'formal_test' or identity['csv_sha256'] != digest(args.captions):
        raise ValueError('Generation used a different CSV or was only a development pilot')
    if identity['source_sha256']['cfg.py'] != digest(args.cfg):
        raise ValueError('Submitted CFG differs from the actual generation implementation')
    if identity['source_sha256']['generate.py'] != digest(ROOT / 'generate.py'):
        raise ValueError('Public generation source differs from the recorded run')
    protocol = read(args.protocol)
    if protocol.get('status') != 'frozen' or identity['protocol']['sha256'] != digest(args.protocol):
        raise ValueError('Generation does not match the frozen protocol')
    if identity['protocol']['variant'] != protocol['main_submission_variant']:
        raise ValueError('Only the method chosen before test evaluation is submitted')
    records = {r['id']: r for r in samples}
    if set(records) != {str(i) for i in range(100)}: raise ValueError('Duplicate or missing generated IDs')
    # Executable paths are intentionally absent from portable identity metadata.
    ffmpeg = args.ffmpeg
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    files = {'cfg.py': args.cfg, 'retrieval.json': args.retrieval}
    for row in rows:
        r = records[row['id']]
        if r['caption_sha256'] != text_sha(row['caption']): raise ValueError('A generated caption was replaced')
        contract = text_sha(canonical({'run_sha256': manifest['run_sha256'], 'id': row['id'],
                                      'caption_sha256': text_sha(row['caption'])}))
        verify_record(args.generation_dir, r, contract)
        path = args.generation_dir / r['mp3']['path']
        if path.name != row['id'] + '.mp3' or path.is_symlink(): raise ValueError('Unexpected MP3 ID/path')
        inspect_mp3(path, ffmpeg, None, identity['settings']['seconds'])
        files['generated_samples/' + path.name] = path
    mp3_hashes = [records[str(i)]['mp3']['sha256'] for i in range(100)]
    if len(set(mp3_hashes)) != 100:
        raise ValueError('Exact duplicate generated files require investigation; no automatic filtering is allowed')
    _, references = reference_inventory(args.reference_dir)
    prediction = read(args.retrieval)
    validate_prediction(prediction, [r['id'] for r in references])
    retrieval_meta = read(args.retrieval.parent / 'retrieval_manifest.json')
    if (retrieval_meta.get('status') != 'retrieval_complete' or retrieval_meta.get('caption_count') != 100
            or retrieval_meta.get('reference_count') != 5367
            or retrieval_meta['model']['checkpoint_sha256'] != CLAP_SHA256
            or retrieval_meta.get('protocol_sha256') != digest(args.protocol)
            or retrieval_meta['model'].get('retrieval_source_sha256') != digest(ROOT / 'retrieve.py')
            or retrieval_meta.get('retrieval_json_sha256') != digest(args.retrieval)
            or retrieval_meta.get('caption_source_sha256') != digest(args.captions)):
        raise ValueError('Retrieval results are not a complete, recorded run with the required model')
    evaluation = read(args.evaluation)
    if (evaluation.get('completed') is not True or evaluation['counts']['generated_clips'] != 100
            or evaluation.get('retrieval_sha256') != digest(args.retrieval)
            or evaluation.get('caption_csv_sha256') != digest(args.captions)):
        raise ValueError('Required metrics are missing or evaluate another input set')
    for name in ['FD-VGG-all-reference', 'FD-VGG-retrieved-proxy',
                 'KL-PASST-softmax-retrieved-proxy', 'CLAP']:
        value = evaluation.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('A required metric is absent or invalid: ' + name)
    if evaluation.get('clap_provenance', {}).get('checkpoint_sha256') != CLAP_SHA256:
        raise ValueError('CLAP evaluation did not use the course-designated checkpoint')
    if evaluation['counts']['fd_reference_clips'] != 5367 or evaluation['counts']['paired_reference_rows'] != 100:
        raise ValueError('Declared metric coverage is incomplete')
    evaluated_hashes = {r['id']: r['source_sha256'] for r in evaluation['clap_audio_inputs']}
    if evaluated_hashes != {i: r['mp3']['sha256'] for i, r in records.items()}:
        raise ValueError('Metrics did not evaluate the actual submitted MP3 files')
    if len(files) != 102 or any(p.is_symlink() or not p.is_file() for p in files.values()):
        raise ValueError('The course ZIP must contain exactly CFG, retrieval JSON and 100 real MP3 files')
    hashes = {name: digest(path) for name, path in files.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hw3-zip-', dir=args.output.parent) as folder:
        archive = Path(folder) / args.output.name
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for name, path in sorted(files.items()): z.write(path, args.student_id + '/' + name)
        checks = archive_check(archive, args.student_id, hashes)
        archive.replace(args.output)
    result = {'archive': str(args.output), 'bytes': args.output.stat().st_size,
              'sha256': digest(args.output), 'student_id': args.student_id,
              'file_sha256': hashes, **checks}
    note = args.output.with_suffix('.checks.json')
    note.write_text(json.dumps(result, indent=2) + '\n')
    return {k: v for k, v in result.items() if k != 'file_sha256'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student-id', default='r14942154')
    p.add_argument('--generation-dir', type=Path, required=True)
    p.add_argument('--retrieval', type=Path, required=True)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--captions', type=Path, default=ROOT / 'sample_code/test.csv')
    p.add_argument('--reference-dir', type=Path, default=ROOT / 'audio_mp3')
    p.add_argument('--cfg', type=Path, default=ROOT / 'cfg.py')
    p.add_argument('--output', type=Path, default=ROOT / 'r14942154.zip')
    p.add_argument('--ffmpeg')
    args = p.parse_args()
    try: print(json.dumps(package(args), indent=2))
    except (ValueError, OSError, KeyError) as error: p.exit(1, 'Not ready to package: ' + str(error) + '\n')


if __name__ == '__main__': main()
