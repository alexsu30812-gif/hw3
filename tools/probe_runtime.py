"""Record HW3 environment and a tiny real-device/MP3 smoke test.

This script never downloads or loads music-model weights and is not generation
or metric evaluation. No credentials or token values are recorded.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_CACHE'] = str(ROOT / 'cache/huggingface/hub')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
os.environ['NUMBA_CACHE_DIR'] = str(ROOT / 'cache/numba')


def sysctl(name):
    try:
        return subprocess.check_output(['sysctl', '-n', name], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        raise SystemExit('Use the isolated hw3/.venv Python')
    versions = {}
    for name in ('torch', 'torchaudio', 'torchvision', 'diffusers', 'transformers',
                 'accelerate', 'torchsde', 'sentencepiece', 'soundfile', 'laion-clap',
                 'librosa', 'scipy', 'numpy', 'huggingface-hub', 'imageio-ffmpeg',
                 'resampy', 'timm', 'passt'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'machine': platform.machine(),
              'macos': platform.mac_ver()[0], 'cpu': sysctl('machdep.cpu.brand_string'),
              'memory_bytes': sysctl('hw.memsize'), 'disk_free_bytes': shutil.disk_usage(ROOT).free,
              'versions': versions, 'imports': {},
              'scope': 'Imports, tiny MPS arithmetic and one-second synthetic tone MP3 roundtrip only. No pretrained model inference.'}
    vendor = ROOT / 'vendor/passt_hear21'
    if (vendor / '.git').exists():
        report['vendor_passt'] = {
            'repository': 'https://github.com/hkchengrex/passt_hear21',
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=vendor, text=True).strip(),
            'distribution_name': 'passt', 'import_name': 'hear21passt'}
    for module, symbol in [('torch', None), ('torchaudio', None), ('torchvision', None),
                           ('diffusers', 'StableAudioPipeline'), ('laion_clap', 'CLAP_Module'),
                           ('hear21passt.base', 'get_basic_model')]:
        try:
            imported = importlib.import_module(module)
            if symbol:
                getattr(imported, symbol)
            report['imports'][module] = {'ok': True}
        except Exception as e:
            report['imports'][module] = {'ok': False, 'error_type': type(e).__name__,
                                          'message': str(e)[:700]}
    try:
        import torch
        mps = {'built': torch.backends.mps.is_built(), 'available': torch.backends.mps.is_available()}
        if mps['available']:
            value = torch.arange(6, dtype=torch.float32, device='mps').reshape(2, 3).sum().item()
            mps.update(tiny_sum=value, tiny_arithmetic_passed=value == 15.)
        report['mps'] = mps
        report['cuda_available'] = torch.cuda.is_available()
    except Exception as e:
        report['device_probe_error'] = {'type': type(e).__name__, 'message': str(e)[:500]}
    try:
        import imageio_ffmpeg
        import numpy as np
        import soundfile as sf
        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
        info = subprocess.check_output([str(ffmpeg), '-version'], text=True).splitlines()[0]
        encoders = subprocess.check_output([str(ffmpeg), '-hide_banner', '-encoders'], text=True,
                                          stderr=subprocess.DEVNULL)
        if 'libmp3lame' not in encoders:
            raise ValueError('The task-local FFmpeg has no libmp3lame encoder')
        folder = ROOT / 'analysis/runtime_smoke'
        folder.mkdir(parents=True, exist_ok=True)
        rate = 44100
        mono = (.1 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype(np.float32)
        stereo = np.column_stack([mono, mono])
        wav, mp3 = folder / 'synthetic_tone.wav', folder / 'synthetic_tone.mp3'
        sf.write(wav, stereo, rate, subtype='PCM_16')
        subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-y', '-i', str(wav),
                        '-codec:a', 'libmp3lame', '-b:a', '192k', str(mp3)], check=True)
        decoded, sr = sf.read(mp3, dtype='float32', always_2d=True)
        pcm = subprocess.check_output([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-i', str(mp3),
                                       '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
        assert sr == rate and decoded.shape[1] == 2 and np.isfinite(decoded).all()
        assert abs(len(decoded) / sr - 1.) < .1 and float(np.max(np.abs(decoded))) > .01
        relative = ffmpeg.relative_to(ROOT) if ffmpeg.is_relative_to(ROOT) else None
        report['ffmpeg'] = {'task_local': relative is not None,
                            'executable_relative_to_hw3': str(relative) if relative else None,
                            'version': info, 'libmp3lame_available': True}
        report['mp3_roundtrip'] = {'passed': True, 'synthetic_not_submission_audio': True,
                                  'sample_rate': sr, 'channels': int(decoded.shape[1]),
                                  'decoded_frames': len(decoded), 'ffmpeg_decoded_frames': len(pcm) // 8,
                                  'duration_seconds': len(decoded) / sr,
                                  'peak': float(np.max(np.abs(decoded))), 'bytes': mp3.stat().st_size}
    except Exception as e:
        report['mp3_roundtrip'] = {'passed': False, 'error_type': type(e).__name__, 'message': str(e)[:700]}
    (ROOT / 'analysis/environment_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
