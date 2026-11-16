"""Verify the real VGGish weights on MPS using a non-submission tone."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import evaluate


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError('The planned MPS feature extractor is unavailable')
    model, provenance = evaluate.load_encoder('vggish', 'mps')
    source = ROOT / 'analysis/runtime_smoke/synthetic_tone.mp3'
    wave, metadata = evaluate.prepared_wave(source, 16000, pad=False)
    wave = wave.repeat(10)  # A ten-second synthetic operator/timing probe only.
    started = time.monotonic()
    with torch.inference_mode():
        values = model(wave.unsqueeze(0))[0].detach().float().cpu()
    if values.ndim != 2 or values.shape[1] != 128 or not torch.isfinite(values).all():
        raise RuntimeError('The real VGGish MPS forward returned invalid features')
    report = {'passed': True, 'synthetic_input_not_submission': True,
              'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': evaluate.digest(source), 'preprocessing': metadata,
              'probe_extension': 'one-second synthetic tone repeated ten times',
              'model_input_frames': int(wave.numel()),
              'feature_shape': list(values.shape), 'forward_seconds': time.monotonic() - started,
              'device': 'mps', 'provenance': provenance,
              'interpretation': 'Operator/device smoke only, not a generation score.'}
    evaluate.write_json(ROOT / 'analysis/vggish_mps_smoke.json', report)
    print(json.dumps({'passed': True, 'device': 'mps', 'feature_shape': list(values.shape)}))


if __name__ == '__main__':
    main()
