"""Check access to required gated weights without exposing credentials.

This performs authenticated metadata/HEAD requests only. It does not accept
terms, register an account, save a token, or download model weights.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone

os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

from huggingface_hub import HfApi, get_hf_file_metadata, get_token, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
REPO = 'stabilityai/stable-audio-open-1.0'
REVISION = 'f21265c1e2710b3bd2386596943f0007f55f802e'
WEIGHT_FILE = 'transformer/diffusion_pytorch_model.safetensors'


def main():
    token = get_token()
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'repo_id': REPO, 'revision': REVISION,
              'credential_available': bool(token), 'terms_accepted_by_agent': False,
              'weight_file_checked': WEIGHT_FILE,
              'authorized_weight_access': False, 'weights_downloaded_by_probe': False}
    try:
        info = HfApi().model_info(REPO, revision=REVISION, token=token)
        result.update(resolved_revision=info.sha, gated=info.gated)
        metadata = get_hf_file_metadata(hf_hub_url(REPO, WEIGHT_FILE, revision=REVISION), token=token)
        result.update(weight_http_status=200, authorized_weight_access=True,
                      weight_bytes=metadata.size, weight_etag=metadata.etag)
    except Exception as error:
        response = getattr(error, 'response', None)
        result.update(error_type=type(error).__name__,
                      weight_http_status=getattr(response, 'status_code', None))
        if response is not None:
            result['error_code'] = response.headers.get('X-Error-Code')
    target = ROOT / 'analysis/model_access_probe.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['authorized_weight_access']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
