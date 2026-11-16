# Sampling-loop portions adapted from Diffusers 0.41.0 StableAudioPipeline:
# https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py
# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use the adapted portions except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Reproducible HW3 generation using our CFG and a locally available model.

Single non-test prompt (weights must already be downloaded legitimately):
  python generate.py --model-path cache/SAO --revision <40-char-commit> \
    --prompt 'A solo acoustic piano playing slow detached notes.' \
    --output-dir results/pilot --device mps --dtype float16 --check

Formal 100-caption generation additionally requires --protocol <frozen.json>.
The supplied sample_code directory is preserved. This loop adapts the course
sample and Diffusers 0.41.0 (Apache-2.0), with CPU RNG, metadata and export checks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import functools
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import random
import shutil
import subprocess
import tempfile
import time

from cfg import guidance_at_step, guided_prediction

ROOT = Path(__file__).resolve().parent
MODEL_REPOSITORY = 'stabilityai/stable-audio-open-1.0'
PINNED_DIFFUSERS = '0.41.0'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False, default=str)


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                    default=str) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_captions(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'caption']:
            raise ValueError('Caption CSV must have exactly id,caption columns')
        rows = list(reader)
    if len(rows) != 100 or {r['id'] for r in rows} != {str(i) for i in range(100)}:
        raise ValueError('Formal CSV must contain each string ID 0..99 exactly once')
    if any(set(row) != {'id', 'caption'} for row in rows):
        raise ValueError('Malformed caption CSV rows must not be silently truncated')
    if any(not isinstance(r['caption'], str) or not r['caption'].strip() for r in rows):
        raise ValueError('Every caption must be nonempty')
    if len({r['caption'] for r in rows}) != 100:
        raise ValueError('Duplicate captions require investigation before generation')
    return sorted(rows, key=lambda r: int(r['id']))


def validate_settings(settings):
    for name in ('guidance_scale', 'end_scale'):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(name + ' must be finite and nonnegative')
    if not math.isfinite(settings['seconds']) or settings['seconds'] <= 0:
        raise ValueError('seconds must be positive and finite')
    if isinstance(settings['steps'], bool) or not isinstance(settings['steps'], int) or settings['steps'] < 1:
        raise ValueError('steps must be a positive integer')
    if not 0 <= settings['rescale'] <= 1 or not math.isfinite(settings['rescale']):
        raise ValueError('rescale must be in [0,1]')
    if settings['schedule'] not in ('constant', 'linear'):
        raise ValueError('Unknown guidance schedule')
    if not isinstance(settings['base_seed'], int) or not 0 <= settings['base_seed'] < 2**63 - 100:
        raise ValueError('base_seed is outside the supported range')
    if not re.fullmatch(r'[0-9a-f]{40}', settings['model_revision']):
        raise ValueError('A pinned 40-character model revision is required')
    if not re.fullmatch(r'(?:128|160|192|224|256|320)k', settings['mp3_bitrate']):
        raise ValueError('Use an explicit supported MP3 bitrate')


def validate_frozen_protocol(path, csv_path, settings):
    protocol = json.loads(Path(path).read_text())
    if protocol.get('status') != 'frozen' or protocol.get('no_test_or_validation_tuning') is not True:
        raise ValueError('Test generation requires a frozen protocol, not the draft')
    try:
        timestamp = datetime.fromisoformat(protocol['frozen_at_utc'])
        if timestamp.tzinfo is None or timestamp > datetime.now(timezone.utc):
            raise ValueError('Protocol timestamp must be timezone-aware and not in the future')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('A valid frozen_at_utc is required') from exc
    if protocol.get('test_csv_sha256') != digest(csv_path):
        raise ValueError('The frozen protocol refers to a different caption CSV')
    variants = protocol.get('generation_conditions', [])
    if not isinstance(variants, list) or not variants or any(not isinstance(v, dict) for v in variants):
        raise ValueError('A nonempty list of generation conditions is required')
    variant_ids = [v.get('id') for v in variants]
    if any(not isinstance(v, str) or not v for v in variant_ids) or len(set(variant_ids)) != len(variant_ids):
        raise ValueError('Generation condition IDs must be unique nonempty strings')
    all_ids = {str(i) for i in range(100)}
    plans = {}
    for variant in variants:
        # Legacy all-caption protocols omit planned_ids; a subset must be
        # explicitly frozen rather than inferred from --max-new at run time.
        planned = variant.get('planned_ids', [str(i) for i in range(100)])
        if (not isinstance(planned, list) or not planned
                or any(not isinstance(i, str) or i not in all_ids for i in planned)
                or len(set(planned)) != len(planned)):
            raise ValueError('planned_ids must be nonempty unique string IDs from 0..99')
        plans[variant['id']] = list(planned)
    main_variant = protocol.get('main_submission_variant')
    if main_variant not in plans:
        raise ValueError('Choose the main submission variant before test generation')
    if set(plans[main_variant]) != all_ids:
        raise ValueError('The main submission variant must plan all 100 caption IDs')
    wanted = canonical(settings)
    matches = [v for v in variants if canonical(v.get('settings')) == wanted]
    if len(matches) != 1 or not matches[0].get('id'):
        raise ValueError('Exact generation settings must appear once in frozen generation_conditions')
    return {'sha256': digest(path), 'frozen_at_utc': protocol['frozen_at_utc'],
            'variant': matches[0]['id'], 'main_submission_variant': main_variant,
            'planned_ids': plans[matches[0]['id']]}


def model_inventory(directory):
    directory = Path(directory).resolve()
    if not (directory / 'model_index.json').is_file():
        raise ValueError('model-path must be a complete local Diffusers model directory')
    allowed = {'.json', '.txt', '.model', '.safetensors', '.bin', '.spm'}
    files = [p for p in sorted(directory.rglob('*')) if p.is_file()
             and p.suffix in allowed
             and p.name != 'model_provenance.json' and not p.name.endswith('.verified.json')
             and not any(part.startswith('.') or part.endswith('.ranges')
                         for part in p.relative_to(directory).parts)]
    if not any(p.suffix in {'.safetensors', '.bin'} for p in files):
        raise ValueError('No local model weights found')
    return {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in files}


def resolve_device(name, dtype_name):
    import torch
    device = name
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS was requested but is unavailable in this process')
    if dtype_name == 'auto':
        dtype_name = 'float16' if device == 'cuda' else 'float32'
    if device == 'cpu' and dtype_name == 'float16':
        raise ValueError('Use float32 for CPU inference')
    return device, dtype_name, getattr(torch, dtype_name)


def reset_sampling_state(pipe, seed):
    """Reset initial/global/Brownian RNG and construct a fresh scheduler per call.

    Some SDE schedulers draw a Brownian seed separately from the Generator passed
    to prepare_latents. Explicitly pin it where supported; apply the same policy
    to our loop and to the native-pipeline comparison.
    """
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    cls = type(pipe.scheduler)
    supports_brownian_seed = 'noise_sampler_seed' in inspect.signature(cls.__init__).parameters
    updates = {'noise_sampler_seed': seed} if supports_brownian_seed else {}
    pipe.scheduler = cls.from_config(dict(pipe.scheduler.config), **updates)
    scheduler = pipe.scheduler
    upstream_source = inspect.getsourcefile(cls)
    compatibility = {'applied': False, 'id': None, 'terminal_only': False,
                     'upstream_scheduler_source_sha256': digest(upstream_source) if upstream_source else None}
    if (cls.__module__ == 'diffusers.schedulers.scheduling_cosine_dpmsolver_multistep'
            and cls.__name__ == 'CosineDPMSolverMultistepScheduler'):
        # Diffusers 0.41.0 requests Brownian noise even for the final sigma=0
        # update. torchsde's tolerance-rounded boundary can recursively split a
        # sub-tolerance interval at that request. The stock first-order update
        # multiplies this noise by sigma_next=0, so its value has no effect.
        # Keep the real scheduler class/config and every nonterminal operation.
        stock_step = scheduler.step

        @functools.wraps(stock_step)
        def terminal_safe_step(model_output, timestep, sample, generator=None, return_dict=True):
            if scheduler.step_index is None:
                scheduler._init_step_index(timestep)
            if float(scheduler.sigmas[scheduler.step_index + 1]) != 0.0:
                return stock_step(model_output, timestep, sample, generator=generator,
                                  return_dict=return_dict)
            previous_sampler = scheduler.noise_sampler
            scheduler.noise_sampler = lambda sigma, sigma_next: torch.zeros_like(model_output)
            try:
                return stock_step(model_output, timestep, sample, generator=generator,
                                  return_dict=return_dict)
            finally:
                scheduler.noise_sampler = previous_sampler

        scheduler.step = terminal_safe_step
        sampler_source = inspect.getsourcefile(inspect.getmodule(cls).BrownianTreeNoiseSampler)
        compatibility.update(applied=True, id='cosine_zero_sigma_noise_guard_v1', terminal_only=True,
            trigger='sigma_next == 0 exactly',
            upstream_brownian_sampler_source_sha256=digest(sampler_source),
            stock_class_and_config_preserved=True, nonterminal_steps_unmodified=True,
            native_comparison_uses_same_guard=True,
            reason='Diffusers 0.41.0 requests terminal Brownian noise outside its configured interval; '
                   'torchsde 0.2.6 can recurse at tolerance-rounded floating-point boundaries.',
            implementation='Per-instance step wrapper temporarily supplies zeros_like(model_output) '
                           'only at exact zero next sigma, calls the stock step/update, and restores '
                           'the previous noise sampler in finally. Its stock noise coefficient is zero.')
    return {'class': cls.__module__ + '.' + cls.__name__,
            'config': dict(pipe.scheduler.config), 'global_seed': seed,
            'noise_sampler_seed': seed if supports_brownian_seed else None,
            'noise_sampler_seed_supported': supports_brownian_seed,
            'compatibility': compatibility}


def generate(pipe, prompt, guidance_scale=7.0, num_inference_steps=100,
             audio_end_in_s=10.0, seed=0, schedule='constant', end_scale=1.0,
             rescale=0.0, return_details=False):
    """Sample both branches at every step, including genuine scale=0/1 endpoints."""
    import torch
    from diffusers.models.embeddings import get_1d_rotary_pos_embed
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must be a nonempty string')
    max_seconds = int(pipe.transformer.config.sample_size) * pipe.vae.hop_length / pipe.vae.config.sampling_rate
    if not math.isfinite(audio_end_in_s) or not 0 < audio_end_in_s <= max_seconds:
        raise ValueError(f'seconds must be in (0, {max_seconds}]')
    if num_inference_steps < 1:
        raise ValueError('At least one inference step is required')
    device = pipe._execution_device
    with torch.inference_mode():
        scheduler_audit = reset_sampling_state(pipe, seed)
        prompt_embeds = pipe.encode_prompt(prompt, device, do_classifier_free_guidance=False)
        seconds_start, seconds_end = pipe.encode_duration(0.0, audio_end_in_s, device, False, batch_size=1)
        cond = torch.cat([prompt_embeds, seconds_start, seconds_end], dim=1)
        encoder_hidden_states = torch.cat([torch.zeros_like(cond), cond])
        global_hidden_states = torch.cat([seconds_start, seconds_end], dim=2).repeat(2, 1, 1)
        # CPU noise is supported by Diffusers' randn_tensor, including MPS.
        generator = torch.Generator('cpu').manual_seed(seed)
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = pipe.prepare_latents(
            1, pipe.transformer.config.in_channels, int(pipe.transformer.config.sample_size),
            cond.dtype, device, generator, audio_channels=pipe.vae.config.audio_channels)
        initial_sha = hashlib.sha256(latents.detach().float().cpu().numpy().tobytes()).hexdigest()
        extra = pipe.prepare_extra_step_kwargs(generator, 0.0)
        rotary = get_1d_rotary_pos_embed(pipe.rotary_embed_dim,
            latents.shape[2] + global_hidden_states.shape[1], use_real=True, repeat_interleave_real=False)
        timesteps = pipe.scheduler.timesteps
        scales = []
        for i, t in enumerate(pipe.progress_bar(timesteps)):
            model_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
            prediction = pipe.transformer(model_input, t.unsqueeze(0),
                encoder_hidden_states=encoder_hidden_states, global_hidden_states=global_hidden_states,
                rotary_embedding=rotary, return_dict=False)[0]
            uncond, conditional = prediction.chunk(2)
            scale = guidance_at_step(guidance_scale, i, len(timesteps), schedule, end_scale)
            prediction = guided_prediction(uncond, conditional, scale, rescale)
            latents = pipe.scheduler.step(prediction, t, latents, **extra).prev_sample
            scales.append(scale)
        audio = pipe.vae.decode(latents).sample[0, :, :int(audio_end_in_s * pipe.vae.config.sampling_rate)]
        if audio.ndim != 2 or not torch.isfinite(audio).all():
            raise ValueError('Generated audio is malformed or non-finite')
        details = {'generator_device': 'cpu', 'initial_latents_sha256_float32': initial_sha,
                   'latent_shape': list(latents.shape), 'guidance_scales': scales,
                   'full_model_latent_length_used': True, 'model_max_seconds': max_seconds,
                   'scheduler': scheduler_audit}
        return (audio, details) if return_details else audio


def find_executable(value, name, required=True):
    path = shutil.which(value) if value else shutil.which(name)
    if path is None and value is None and name == 'ffmpeg':
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if path is None and value and Path(value).is_file():
        path = str(Path(value).resolve())
    if not path:
        if value:
            raise ValueError('Explicit executable path is unavailable: ' + name)
        if not required:
            return None, None
        raise ValueError(name + ' is required; provide a task-local binary path')
    version = subprocess.check_output([path, '-version'], text=True).splitlines()[0]
    return str(Path(path).resolve()), version


def inspect_mp3(path, ffmpeg, ffprobe, expected_seconds, rate=44100):
    import numpy as np
    import soundfile as sf
    if ffprobe:
        info = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams',
            '-show_format', '-of', 'json', str(path)], text=True))
        streams = [s for s in info['streams'] if s.get('codec_type') == 'audio']
        if len(streams) != 1 or streams[0].get('codec_name') != 'mp3':
            raise ValueError('Output must contain one genuine MP3 audio stream')
        stream = streams[0]
        if int(stream['sample_rate']) != rate or int(stream['channels']) != 2:
            raise ValueError('Submitted MP3 has wrong rate/channels')
        probe = 'ffprobe plus full ffmpeg decode'
    else:
        # libsndfile 1.2+ identifies/reads real MPEG audio; extension alone is not
        # evidence of MP3. ffmpeg below independently decodes every sample.
        info = sf.info(path)
        if info.format not in ('MP3', 'MPEG-1/2 Audio') or info.samplerate != rate or info.channels != 2:
            raise ValueError('libsndfile did not identify 44.1kHz stereo MP3')
        probe = 'libsndfile format probe plus full ffmpeg decode'
    pcm = subprocess.check_output([ffmpeg, '-v', 'error', '-i', str(path), '-map', '0:a:0',
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
    audio = np.frombuffer(pcm, dtype='<f4')
    if not len(audio) or len(audio) % 2 or not np.isfinite(audio).all():
        raise ValueError('MP3 decode failed, is empty, or contains non-finite samples')
    audio = audio.reshape(-1, 2)
    seconds = len(audio) / rate
    rms = float(np.sqrt(np.mean(audio.astype('float64') ** 2)))
    if abs(seconds - expected_seconds) > 0.1 or rms <= 1e-6:
        raise ValueError('MP3 is truncated/too long or effectively silent')
    return {'sample_rate': rate, 'channels': 2, 'decoded_frames': len(audio),
            'decoded_seconds': seconds, 'rms': rms, 'peak': float(np.abs(audio).max()),
            'fraction_abs_ge_1': float(np.mean(np.abs(audio) >= 1.0)), 'codec': 'mp3',
            'validation_method': probe}


def verify_record(directory, record, expected_contract):
    directory = Path(directory)
    if record.get('contract_sha256') != expected_contract:
        raise ValueError('Existing output belongs to another model, prompt or configuration')
    for kind in ('wav', 'mp3'):
        name = record[kind]['path']
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe recorded output path')
        path = directory / rel
        if path.is_symlink() or not path.is_file() or digest(path) != record[kind]['sha256']:
            raise ValueError('An existing output changed; preserve and investigate: ' + name)


def write_audio(directory, identifier, audio, rate, expected_seconds, bitrate, binaries):
    import numpy as np
    import soundfile as sf
    directory = Path(directory)
    array = audio.detach().float().cpu().numpy().T
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError('Expected finite stereo generated audio')
    if len(array) != int(expected_seconds * rate) or float(np.mean(array.astype('float64') ** 2)) <= 1e-12:
        raise ValueError('Generated audio has incorrect length or no signal')
    names = {'wav': f'masters/{identifier}.wav', 'mp3': f'generated_samples/{identifier}.mp3'}
    for name in names.values():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError('Orphan/unrecorded audio is preserved, not overwritten: ' + str(path))
    with tempfile.TemporaryDirectory(prefix='.piece-', dir=directory) as temporary:
        wav, mp3 = Path(temporary) / 'audio.wav', Path(temporary) / 'audio.mp3'
        sf.write(wav, array, rate, subtype='FLOAT')
        subprocess.run([binaries['ffmpeg']['path'], '-nostdin', '-v', 'error', '-y', '-i', str(wav),
            '-map_metadata', '-1', '-c:a', 'libmp3lame', '-b:a', bitrate, '-ar', str(rate),
            '-ac', '2', '-id3v2_version', '3', str(mp3)], check=True)
        audit = inspect_mp3(mp3, binaries['ffmpeg']['path'], binaries['ffprobe']['path'], expected_seconds, rate)
        result = {kind: {'path': names[kind], 'sha256': digest(path), 'bytes': path.stat().st_size}
                  for kind, path in [('wav', wav), ('mp3', mp3)]}
        wav.replace(directory / names['wav'])
        mp3.replace(directory / names['mp3'])
    result['audio_audit'] = audit
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt', help='Independent non-test pilot caption')
    source.add_argument('--captions', type=Path, help='Formal 100-caption CSV; requires frozen protocol')
    parser.add_argument('--id', default='pilot')
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--dtype', choices=['auto', 'float16', 'float32'], default='auto')
    parser.add_argument('--scale', type=float, default=7.)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--base-seed', type=int, default=20261115)
    parser.add_argument('--schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--end-scale', type=float, default=1.)
    parser.add_argument('--rescale', type=float, default=0.)
    parser.add_argument('--mp3-bitrate', default='192k')
    parser.add_argument('--ffmpeg')
    parser.add_argument('--ffprobe')
    parser.add_argument('--max-new', type=int,
                        help='Safe per-invocation cap on new pieces within the frozen planned IDs')
    parser.add_argument('--check', action='store_true', help='Compare vanilla scale>1 against official Diffusers')
    parser.add_argument('--check-atol', type=float, default=1e-4)
    parser.add_argument('--print-settings', action='store_true',
                        help='Print the resolved settings object for a future frozen protocol, without loading weights')
    args = parser.parse_args()
    lock_handle = None
    try:
        device, dtype_name, dtype = resolve_device(args.device, args.dtype)
        settings = {'model_repository': MODEL_REPOSITORY, 'model_revision': args.revision,
            'guidance_scale': args.scale, 'steps': args.steps, 'seconds': args.seconds,
            'base_seed': args.base_seed, 'schedule': args.schedule, 'end_scale': args.end_scale,
            'rescale': args.rescale, 'device': device, 'dtype': dtype_name, 'generator_device': 'cpu',
            'scheduler_rng_policy': 'fresh scheduler per piece; seed global RNG and noise_sampler_seed when supported',
            'mp3_bitrate': args.mp3_bitrate}
        validate_settings(settings)
        if args.max_new is not None and args.max_new < 1:
            raise ValueError('max-new must be positive')
        if args.check and (args.scale <= 1 or args.schedule != 'constant' or args.rescale != 0):
            raise ValueError('Official parity requires vanilla constant CFG at scale > 1')
        if not math.isfinite(args.check_atol) or args.check_atol < 0:
            raise ValueError('check-atol must be finite and nonnegative')
        if args.print_settings:
            print(json.dumps(settings, indent=2))
            return
        if args.captions:
            if args.protocol is None:
                raise ValueError('Formal test generation requires --protocol with a frozen decision log')
            rows = read_captions(args.captions)
            protocol = validate_frozen_protocol(args.protocol, args.captions, settings)
            mode, csv_sha = 'formal_test', digest(args.captions)
        else:
            if not args.prompt.strip() or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.id):
                raise ValueError('Pilot needs a nonempty prompt and a safe identifier')
            supplied = ROOT / 'sample_code/test.csv'
            if supplied.is_file() and args.prompt in {r['caption'] for r in read_captions(supplied)}:
                raise ValueError('This is a supplied test caption; use the frozen formal protocol')
            rows, protocol, mode, csv_sha = [{'id': args.id, 'caption': args.prompt}], None, 'development_pilot', None
        planned_ids = set(protocol['planned_ids']) if protocol else {args.id}
        # Preserve all formal prompts in the run identity while limiting the
        # work itself to the predeclared cohort, including after a restart.
        planned_rows = [row for row in rows if row['id'] in planned_ids]
        binaries = {}
        for name in ('ffmpeg', 'ffprobe'):
            executable, version = find_executable(getattr(args, name), name, required=name == 'ffmpeg')
            binaries[name] = {'path': executable, 'version': version,
                              'sha256': digest(executable) if executable else None}
        if importlib.metadata.version('diffusers') != PINNED_DIFFUSERS:
            raise ValueError('Generation requires diffusers==' + PINNED_DIFFUSERS)
        versions = {n: importlib.metadata.version(n) for n in
                    ('torch', 'torchaudio', 'diffusers', 'transformers', 'accelerate', 'soundfile', 'numpy')}
        print('Hashing the locally available model; no model download is performed.', flush=True)
        inventory = model_inventory(args.model_path)
        identity = {'settings': settings, 'mode': mode, 'csv_sha256': csv_sha, 'protocol': protocol,
            'model_files': inventory, 'runtime_versions': versions,
            'official_parity_policy': {'requested': args.check,
                                       'absolute_tolerance': args.check_atol if args.check else None},
            'source_sha256': {'generate.py': digest(__file__), 'cfg.py': digest(ROOT / 'cfg.py')},
            'codecs': {n: {k: b[k] for k in ('version', 'sha256')} for n, b in binaries.items()},
            'prompts': [{'id': r['id'], 'caption_sha256': text_sha(r['caption'])} for r in rows]}
        run_sha = text_sha(canonical(identity))
        if args.output_dir.is_symlink():
            raise ValueError('Output directory must not be a symlink')
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('.generation.lock', 'manifest.json', 'masters', 'generated_samples', 'records'):
            if (directory / name).is_symlink():
                raise ValueError('Output/control paths must not be symlinks: ' + name)
        import fcntl
        lock_handle = (directory / '.generation.lock').open('a')
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another generator is writing this output directory') from exc
        manifest_path = directory / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('run_sha256') != run_sha:
                raise ValueError('Output directory belongs to another exact configuration/runtime/input')
        else:
            if any(p.name != '.generation.lock' for p in directory.iterdir()):
                raise ValueError('Use an empty output directory or one with its matching manifest')
            manifest = {'run_sha256': run_sha, 'identity': identity, 'started_at_utc': utc_now(),
                        'complete': False, 'formal_complete': False, 'samples': [],
                        'planned_count': len(planned_ids), 'planned_complete': False}
            write_json(manifest_path, manifest)
        records = {r['id']: r for r in manifest['samples']}
        if len(records) != len(manifest['samples']) or set(records) - planned_ids:
            raise ValueError('Manifest contains duplicated or unplanned IDs')
        contracts = {r['id']: text_sha(canonical({'run_sha256': run_sha, 'id': r['id'],
                      'caption_sha256': text_sha(r['caption'])})) for r in rows}
        record_dir = directory / 'records'
        record_dir.mkdir(exist_ok=True)
        if any(p.stem not in planned_ids for p in record_dir.glob('*.json')):
            raise ValueError('Per-piece records contain unplanned IDs; preserve and investigate')
        for folder, suffix in [('masters', 'wav'), ('generated_samples', 'mp3')]:
            allowed = {identifier + '.' + suffix for identifier in planned_ids}
            present = {p.name for p in (directory / folder).glob('*') if p.is_file()}
            if present - allowed:
                raise ValueError('Unexpected audio files are preserved, not submitted: ' +
                                 ', '.join(sorted(present - allowed)))
        for row in planned_rows:
            identifier = row['id']
            record_path = record_dir / (identifier + '.json')
            if identifier not in records and record_path.exists():
                recovered = json.loads(record_path.read_text())
                if recovered.get('id') != identifier:
                    raise ValueError('Per-piece record ID changed')
                records[identifier] = recovered
            if identifier in records:
                verify_record(directory, records[identifier], contracts[identifier])
                if record_path.exists() and json.loads(record_path.read_text()) != records[identifier]:
                    raise ValueError('Per-piece metadata differs from the manifest: ' + identifier)
            elif any((directory / f'{folder}/{identifier}.{suffix}').exists()
                     for folder, suffix in [('masters', 'wav'), ('generated_samples', 'mp3')]):
                raise ValueError('Unrecorded output preserved; inspect orphan ID ' + identifier)
        if set(records) != planned_ids:
            import torch
            from diffusers import StableAudioPipeline
            print(json.dumps({'loading_local_model': MODEL_REPOSITORY, 'revision': args.revision,
                              'device': device, 'dtype': dtype_name}), flush=True)
            pipe = StableAudioPipeline.from_pretrained(str(args.model_path), torch_dtype=dtype,
                                                       local_files_only=True).to(device)
            for component in ('text_encoder', 'projection_model', 'transformer', 'vae'):
                getattr(pipe, component).eval()
            sample_rate = int(pipe.vae.config.sampling_rate)
            if sample_rate != 44100 or int(pipe.vae.config.audio_channels) != 2:
                raise ValueError('Expected the Stable Audio Open 44.1kHz stereo model')
            new_count = 0
            for row in planned_rows:
                identifier = row['id']
                if identifier in records:
                    continue
                if args.max_new is not None and new_count >= args.max_new:
                    break
                seed = args.base_seed + (int(identifier) if mode == 'formal_test' else 0)
                began = time.monotonic()
                tokens = pipe.tokenizer(row['caption'], truncation=False).input_ids
                token_audit = {'untruncated_tokens': len(tokens),
                              'maximum_tokens': int(pipe.tokenizer.model_max_length),
                              'truncated': len(tokens) > pipe.tokenizer.model_max_length}
                audio, details = generate(pipe, row['caption'], args.scale, args.steps, args.seconds, seed,
                    args.schedule, args.end_scale, args.rescale, return_details=True)
                parity = None
                if args.check:
                    reset_sampling_state(pipe, seed)
                    reference = pipe(row['caption'], num_inference_steps=args.steps,
                        audio_end_in_s=args.seconds, guidance_scale=args.scale,
                        generator=torch.Generator('cpu').manual_seed(seed)).audios[0]
                    if reference.shape != audio.shape:
                        raise ValueError('Official and custom waveform shapes differ')
                    difference = (reference.float() - audio.float()).abs()
                    maximum = float(difference.max().item())
                    parity = {'max_abs_difference': maximum, 'mean_abs_difference': float(difference.mean().item()),
                              'absolute_tolerance': args.check_atol, 'passed': maximum <= args.check_atol}
                    if not parity['passed']:
                        write_json(directory / 'parity_failure.json', {'id': identifier, 'parity': parity})
                        raise ValueError('CFG waveform parity failed; no completed audio was committed')
                outputs = write_audio(directory, identifier, audio, sample_rate, args.seconds,
                                      args.mp3_bitrate, binaries)
                record = {'id': identifier, 'caption_sha256': text_sha(row['caption']), 'seed': seed,
                    'contract_sha256': contracts[identifier], 'completed_at_utc': utc_now(),
                    'generation_seconds': time.monotonic() - began, 'tokenization': token_audit,
                    'sampling': details, 'official_waveform_parity': parity, **outputs}
                write_json(record_dir / (identifier + '.json'), record)
                records[identifier] = record
                manifest.update(samples=[records[r['id']] for r in rows if r['id'] in records],
                                completed_count=len(records), planned_count=len(planned_ids),
                                planned_complete=set(records) == planned_ids, updated_at_utc=utc_now())
                write_json(manifest_path, manifest)
                new_count += 1
                print(json.dumps({'completed': identifier, 'count': len(records), 'target': len(planned_ids),
                                  'full_caption_count': len(rows),
                                  'seconds': record['generation_seconds']}), flush=True)
        manifest.update(samples=[records[r['id']] for r in rows if r['id'] in records],
                        completed_count=len(records), complete=len(records) == len(rows),
                        formal_complete=mode == 'formal_test' and len(records) == 100,
                        planned_count=len(planned_ids), planned_complete=set(records) == planned_ids,
                        updated_at_utc=utc_now())
        write_json(manifest_path, manifest)
        print(json.dumps({'manifest': str(manifest_path), 'complete': manifest['complete'],
                          'formal_complete': manifest['formal_complete'], 'pieces': len(records),
                          'planned_count': manifest['planned_count'],
                          'planned_complete': manifest['planned_complete']}, indent=2))
    except (ValueError, OSError, RuntimeError, ImportError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(1, 'Generation stopped: ' + str(exc) + '\n')
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == '__main__':
    main()
