"""Run real VGGish/PaSST weights on a synthetic, non-submission MP3 only."""
from __future__ import annotations

import concurrent.futures
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback

import certifi
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['HF_HUB_CACHE'] = str(ROOT / 'cache/huggingface/hub')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')
spec = importlib.util.spec_from_file_location('hw3_evaluator_under_review', ROOT / 'evaluate.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / 'analysis/runtime_smoke/synthetic_tone.mp3'
    if not source.is_file():
        raise SystemExit('Run tools/probe_runtime.py first for the synthetic input')
    report = {'status': 'running', 'synthetic_input_not_submission': True,
              'source_mp3': str(source.relative_to(ROOT)), 'source_sha256': digest(source),
              'adapter_sha256_before': digest(ROOT / 'evaluate.py'),
              'scope': 'Real pretrained encoder smoke on one-second synthetic tone, not test/reference features or HW3 metrics.',
              'tls_ca': 'certifi certificate bundle', 'encoders': {}}
    target = ROOT / 'analysis/evaluator_encoder_smoke.json'
    def save():
        target.write_text(json.dumps(report, indent=2) + '\n')
    def load(kind):
        begin = time.monotonic()
        print(json.dumps({'loading_encoder': kind}), flush=True)
        model, provenance = evaluator.load_encoder(kind, 'cpu')
        return model, provenance, time.monotonic() - begin
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {kind: pool.submit(load, kind) for kind in ('vggish', 'passt')}
        for kind, future in futures.items():
            try:
                model, provenance, load_seconds = future.result()
                wave, preprocessing = evaluator.prepared_wave(source, 16000 if kind == 'vggish' else 32000,
                                                               pad=kind == 'passt')
                began = time.monotonic()
                with torch.inference_mode():
                    result = model(wave.unsqueeze(0))
                values = result[0].detach().float().cpu().numpy()
                if kind == 'passt':
                    values = values[:527]
                expected = values.ndim == 2 and values.shape[1] == 128 if kind == 'vggish' else values.shape == (527,)
                if not expected or not np.isfinite(values).all():
                    raise ValueError(f'Invalid {kind} feature shape/values: {values.shape}')
                report['encoders'][kind] = {'passed': True, 'shape': list(values.shape),
                    'all_finite': True, 'feature_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
                    'preprocessing': preprocessing, 'provenance': provenance,
                    'load_and_download_seconds': load_seconds, 'forward_seconds': time.monotonic() - began}
                del model, result, values
                gc.collect()
            except Exception as e:
                report['encoders'][kind] = {'passed': False, 'error_type': type(e).__name__,
                                            'message': str(e)[:1500], 'traceback': traceback.format_exc()}
            save()
            print(json.dumps({'encoder': kind, 'passed': report['encoders'][kind]['passed']}), flush=True)
    report['adapter_sha256_after'] = digest(ROOT / 'evaluate.py')
    report['adapter_unchanged_during_probe'] = report['adapter_sha256_before'] == report['adapter_sha256_after']
    report['status'] = 'passed' if all(x['passed'] for x in report['encoders'].values()) else 'failed'
    save()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
