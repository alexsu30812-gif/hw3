# Portable inference and artifact layout

This folder contains source and recorded results, not the released dataset or
model caches. Supply the course caption CSV and MusicCaps audio yourself. No
model was trained for this assignment; the gated pretrained Stable Audio Open
weights must be obtained from their official pinned revision after personally
accepting its terms. SAO/CLAP downloads verify pinned HF LFS full hashes;
evaluation downloads verify pinned mirror SHA256 and official release size,
because an independent full digest of the old official release is unavailable. Public
model provenance is under `model_provenance/`; no authentication token is bundled.

Install `requirements.txt` in a Python 3.11 virtual environment. For the report,
use a separate environment with `requirements-report.txt` and a local Traditional
Chinese font. Download models with `tools/prepare_sao.py --download` and
`tools/download_clap.py --download` after local HF login where needed.

Pass `--captions /path/to/your/course_captions.csv` explicitly to generation,
retrieval, analysis and report commands; pass `--audio-dir /path/to/musiccaps`
to retrieval/extraction. The supplied CSV must match the SHA recorded in the
frozen protocol. The CSV is intentionally absent from this public folder.
Generation also needs `--protocol analysis/experiment_protocol_frozen.json`
and the exact condition settings recorded there. Set a NEW output directory,
such as `reproduced/baseline`; the compact recorded results omit WAV masters
and are evidence, not resumable working caches. Use the original device/runtime
for the closest reproducibility; same seeds do not promise cross-device byte
equality. Do not overwrite the frozen study to select a method using test scores.

The official COOL ZIP is a separate deliverable. This source archive is for the
public project folder and is not the file uploaded to the COOL ZIP slot.

---

This bundle has **full_with_ablation** scope, including the
predeclared four-condition paired 20-caption comparison.

# HW3 — Text-to-Music Generation and Evaluation

Student ID: **r14942154**

## Current status

The recorded local experiments are complete: **100 baseline music clips**,
**20 clips for each of three extra CFG conditions** (160 MP3 files total),
CLAP retrieval for all 100 captions against 5367 MusicCaps references, and
the required full-100 evaluation. The paired 2×2 ablation covers the same
predeclared 20 caption IDs in four conditions; it does not replace the full
100-clip evaluation. The baseline was selected before test scores.

| Required evaluation / explicitly labeled reference policy | Recorded score |
| --- | ---: |
| FD-VGG-all-reference | 7.690776 |
| FD-VGG-retrieved-proxy | 2.527220 |
| KL-PASST-softmax-retrieved-proxy | 1.074031 |
| CLAP | 0.341671 |

Scores are read from the completed experiment JSON, not estimates. FD against
all 5367 references and FD against 100 retrieved rows are different populations.
KL is **KL(retrieved reference || generated)**: retrieved audio is a proxy,
not ground truth for the new caption. Repeated retrieved references are counted
as paired rows; the unique count is reported in the evaluation JSON. The TA's
intended reference policy remains unconfirmed. CLAP measures global alignment,
and none of these metrics establishes human musical preference. **MQAScore was
not implemented or claimed.** No human listening is implied by signal metrics.

Public artifact destination: https://github.com/alexsu30812-gif/hw3
GitHub substitution for the requested cloud-drive folder remains unconfirmed
by the TA. Local finalization and publication verification are separately
recorded in `analysis/finalization_state.json`; this destination alone does not
claim that upload has succeeded.

The course deadline is **2026-11-16 23:59 GMT+8** (2026-11-16 08:59 Phoenix).
The late acceptance window is not an extension.

## Environment

Use Python 3.11 and an isolated environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The original course requirements remain in `sample_code/requirements.txt`.
The root requirements preserve its generation versions, add the required
retrieval/evaluation packages and install the reviewed PaSST source under
`vendor/passt_hear21`. `requirements-lock.txt` records the installed versions.
ImageIO supplies a task-local FFmpeg binary with libmp3lame, so no global codec
installation is required. Runtime probes are in `analysis/environment_report.json`.

## Model access and fixed sources

Stable Audio Open requires the user's own acceptance of the model terms at
<https://huggingface.co/stabilityai/stable-audio-open-1.0> and local login:

```bash
.venv/bin/hf auth login --format human --no-add-to-git-credential
.venv/bin/python tools/check_model_access.py
.venv/bin/python tools/prepare_sao.py --download
```

Credentials stay in the official local HF store and are never included in
artifacts. `prepare_sao.py` downloads the Diffusers components, including the
projection model, at commit `f21265c1e2710b3bd2386596943f0007f55f802e` and checks
their official Git/LFS hashes. It excludes duplicate root checkpoint formats.

The required CLAP file is **music_audioset_epoch_15_esc_90.14.pt**, with
**HTSAT-base / fusion=False**. Its full SHA256 is
`fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd`.
Do not substitute the av-benchmark default music-speech checkpoint.

```bash
.venv/bin/python tools/download_clap.py --download
```

`tools/download_ranges.py` is a resumable alternative for slow connections;
no file is published to its destination until its entire official SHA matches.
Working caches and model download chunks are excluded from submission.

## CFG and an independent pilot

The baseline in `cfg.py` is
`unconditional + scale * (conditional - unconditional)`. The model predicts v,
despite the sample variable name `noise_pred`. Optional linear guidance and
per-item prediction rescaling are explicit experimental switches, not assumed
quality improvements.

After the model download completes:

```bash
.venv/bin/python generate.py \
  --model-path cache/models/stable-audio-open-1.0-f21265c1 \
  --revision f21265c1e2710b3bd2386596943f0007f55f802e \
  --prompt "A dry solo acoustic piano playing slow detached notes, without speech." \
  --output-dir results/development/piano_baseline \
  --device mps --dtype float16 --steps 100 --seconds 10 --scale 7 --check
```

Use `scale>1` for the native-pipeline comparison: Diffusers disables CFG at
scale≤1, whereas the exercise formula still has genuine unconditional and
conditional endpoints. `--check` resets RNG/scheduler state for both calls and
records the actual waveform difference. No successful pretrained parity is
claimed until this real check runs. A zero-weight architecture timing probe is
only an operator/memory check, not trained-model generation.

## Formal runs and reproducibility

Formal generation uses `--captions /path/to/your/course_captions.csv --protocol <frozen.json>`.
The protocol must bind the original CSV SHA and exact `--print-settings` output,
choose the submitted method **before** test scores, and record a UTC freeze
time. `analysis/experiment_protocol_draft.json` intentionally cannot authorize
test generation. Independent development prompts can check runtime and choose
settings; the supplied validation/test examples cannot be used for tuning or
per-caption seed search.

Each ID has a fixed CPU-initial-noise seed shared across experimental variants.
The generator records model/config hashes, package versions, device/dtype,
scheduler, initial-noise hash, original caption hash, codec and output SHA.
It writes real WAV masters and MP3 results atomically, and refuses changed or
foreign artifacts during resume. Same seeds do not promise bitwise equality
across different hardware/software stacks.

For a CUDA/CPU/MPS reproduction of the preselected method, use `reproduce.py`.
It checks the original protocol, caption CSV, complete baseline manifest,
source and local model hashes. Only device/dtype can change; every algorithm,
seed, duration, step count and MP3 setting comes from the original run:

```bash
.venv/bin/python reproduce.py \
  --protocol analysis/experiment_protocol_frozen.json \
  --manifest results/baseline/manifest.json --captions /path/to/original_test.csv \
  --model-path cache/models/stable-audio-open-1.0-f21265c1 \
  --output-dir reproduced/baseline --device cuda --dtype float16
```

The derived protocol preserves the original method-freeze time and separately
records the hardware migration; it is not a new method selected from test
scores. Use a new output directory. Cross-device byte equality is not promised.
The compact public bundle omits the original CSV and working top-k audit,
which contains all captions. Supply the course data locally and rerun retrieval
to recreate that audit if rebuilding the report.

To reproduce retrieval too, add both `--reference-audio-dir /path/to/audio_mp3`
and `--clap-checkpoint /path/to/music_audioset_epoch_15_esc_90.14.pt`. The wrapper
verifies the original 5367-file inventory and fixed CLAP source/policy. It uses
a separate cache and only changes the retrieval device; CLAP remains float32.
Very close cosine rankings can vary across hardware.

Retrieval likewise requires a frozen source/preprocessing/input manifest.
Audio is channel-averaged, centered, resampled to 48kHz, cropped/padded to
exactly 480000 samples and quantized once by the official CLAP API. This removes
implicit random crops. Reference IDs are sorted lexically; exact cosine ties
use that order. The original captions are retained and actual 77-token
truncation is reported. Retrieval returns string keys `"0"…"99"` and actual
MusicCaps filename stems. No integer file-list positions or similarities are
submitted instead of IDs.

## Required evaluation

Prepare the two feature encoders with:

```bash
.venv/bin/python tools/prepare_evaluation_models.py --download --kind all
```

This verifies full file sizes and pinned mirror LFS SHA256 before loading.
The mirrors match the original VGGish/PaSST release sizes; VGGish also matches
the official filename hash prefix. Those older GitHub releases publish no
full asset digest, so the mirror hashes are labeled as transport checks,
not independent official full-digest verification. Truncated prior downloads
are quarantined rather than loaded. Model architecture and metric formulas
are unchanged.

`evaluate.py` imports the original av-benchmark metric functions and lazily
loads only VGGish, PaSST and the specified CLAP. The reviewed source commit is
`f351b9a6fc6abde746d5f8e1d4c47c883319cb41`; PaSST is
`1ab778984f67cca09dd8bbad11f728a0122cce24`. The adapter adds MP3 decoding,
device portability and strict coverage, and avoids unrelated video encoders.

The supplied instructions do not specify the reference population/pairing.
Accordingly, we must explicitly distinguish:

- FD-VGG using the complete 5367-clip reference population.
- A separately labeled FD using the 100 retrieved proxy rows.
- Paired KL-PaSST-softmax, **KL(reference || generated)**, using a stated mapping.
  A retrieval-derived reference is a proxy, not ground-truth music for the new text.
- CLAP against the **original captions** and the actual submitted MP3 files.

TA confirmation of the intended reference policy remains unavailable. No
missing caption, unmatched pair, invalid score, stale cache or upstream failure
sentinel counts as successful evaluation. Reference/generated feature versions
must agree, and every metric input hash is checked against the submitted audio.

CLI overview:

```bash
.venv/bin/python evaluate.py --help
.venv/bin/python evaluate.py extract --help
.venv/bin/python evaluate.py score --help
```

MQAScore is optional. It requires concept-level binary answer probabilities
following its paper, plus reliability checks. Free-text audio descriptions or
self-reported confidence are not MQAScore or human listening evidence.

## Deliverables

The final report must be `r14942154_report.pdf` (16:9), containing actual
methods/results, extra experiments, analysis, limitations and citations.
Metric performance itself is explicitly not the HW3 grading criterion.

The ten-page analysis revision uses `tools/build_report_analysis.py`; it reads
the same real outputs and frozen protocol, adds the existing cohort FD,
per-caption directional counts, both Combined extremes, and result-dependent
discussion. It does not change inference or metric computation. The supervisor's
full finalizer builds this revision automatically after all experiments pass.
To rebuild it manually in the report environment:

```bash
.venv_report/bin/python tools/build_report_analysis.py --artifact-url https://github.com/alexsu30812-gif/hw3
```

The command refuses pending or inconsistent ablation results. Use `--draft`
and an output name containing `DRAFT` only for an explicitly incomplete preview.
The original `tools/build_report.py` remains unchanged for the baseline report.

The separate COOL ZIP must be:

```text
r14942154.zip
└── r14942154/
    ├── cfg.py
    ├── retrieval.json
    └── generated_samples/
        ├── 0.mp3
        ├── ...
        └── 99.mp3
```

`package_submission.py` refuses incomplete/pilot runs, changed source/audio,
incorrect retrieval values or missing required metric coverage; it reads back
every ZIP member and hash. No final ZIP is fabricated while outputs are missing.
Code, README, requirements and the appropriate model provenance go in a public
cloud-drive folder, whose link belongs in the report and COOL comments.
GitHub substitution has not been confirmed by the TA. Never include the
released dataset, model caches, credentials or unrelated checkpoints.

## Sources

- Provided `Homework_3.pdf`, course sample code and COOL screenshot.
- Stable Audio Open model/card/license: <https://huggingface.co/stabilityai/stable-audio-open-1.0>.
- Diffusers 0.41.0 Stable Audio pipeline, Apache-2.0:
  <https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py>.
- LAION-CLAP, CC0-1.0: <https://github.com/LAION-AI/CLAP>.
- av-benchmark, MIT: <https://github.com/hkchengrex/av-benchmark>.
- PaSST, Apache-2.0: <https://github.com/hkchengrex/passt_hear21>.
- MQAScore paper: <https://arxiv.org/abs/2610.31083>.

AI assistance is used for implementation and analysis, with provenance and
limitations disclosed. The supplied HW3 PDF permits cited public code and
pretrained models; it does not contain the separate HW2 AI-permission wording.
No model score, listening result or actual grade is invented.
