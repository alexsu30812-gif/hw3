"""Build the analysis revision of the plain Chinese 16:9 HW3 report.

No score is imputed. Final mode rejects missing/inconsistent results or a missing
artifact URL. --draft visibly marks every page and cannot use the official name.
Run with .venv_report/bin/python; generation/evaluation environments stay separate.
The original frozen tools/build_report.py is preserved. This revision changes
presentation and descriptive analysis only, never generation or metric methods.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
STUDENT = 'r14942154'
CLAP_SHA = 'fae3e9c087f2909c28a09dc31c8dfcdacbc42ba44c70e972b58c1bd1caf6dedd'
KL = 'KL-PASST-softmax-retrieved-proxy'
WIDTH, HEIGHT = 1120, 630
LEFT, RIGHT, CONTENT_BOTTOM = 54, 1066, 58
FONT = 'HW3-CJK'
INK = colors.HexColor('#243b53')
BODY = colors.HexColor('#202b36')
PALE = colors.HexColor('#edf2f6')
LABELS = {'baseline': 'Baseline', 'schedule_only': '只改排程',
          'rescale_only': '只做幅度修正', 'combined': '兩者一起'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value, digits=4):
    return f'{value:.{digits}f}' if finite(value) else '待完成'


def interval(value):
    if not isinstance(value, dict):
        return '待完成'
    bounds = value.get('percentile_95_ci', [])
    if len(bounds) != 2 or not all(finite(v) for v in bounds):
        return '待完成'
    return f"{number(value.get('mean'))} [{number(bounds[0])}, {number(bounds[1])}]"


def read_optional(path, reasons, label):
    if path is None or not Path(path).is_file():
        reasons.append(label + '尚未產生')
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (ValueError, OSError) as exc:
        reasons.append(label + '無法讀取：' + str(exc))
        return {}


def caption_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['id', 'caption']:
            raise ValueError('Caption CSV must have exactly id,caption columns')
        rows = list(reader)
    if len(rows) != 100 or {r['id'] for r in rows} != {str(i) for i in range(100)}:
        raise ValueError('Caption IDs must be exactly 0..99')
    if any(not isinstance(r['caption'], str) or not r['caption'].strip() for r in rows):
        raise ValueError('Every original caption must be nonempty')
    return {r['id']: r['caption'] for r in rows}


def validate_bundle(args):
    reasons = []
    paths = {'protocol': args.protocol, 'generation': args.generation_dir / 'manifest.json',
             'evaluation': args.evaluation,
             'retrieval_manifest': args.retrieval_dir / 'retrieval_manifest.json',
             'retrieval': args.retrieval_dir / 'retrieval.json',
             'retrieval_details': args.retrieval_dir / 'topk_and_tokenization.json'}
    if not getattr(args, 'baseline_only', False):
        paths['ablation'] = args.ablation
    data = {k: read_optional(v, reasons, k) for k, v in paths.items()}
    data.setdefault('ablation', {})
    data['captions'] = caption_rows(args.captions)
    protocol, gen, evaluation, ablation = [data[k] for k in ('protocol', 'generation', 'evaluation', 'ablation')]
    retrieval, meta, details = [data[k] for k in ('retrieval', 'retrieval_manifest', 'retrieval_details')]
    expected = {str(i) for i in range(100)}
    csv_sha = sha(args.captions)

    def check(condition, message):
        if not condition:
            reasons.append(message)

    check(protocol.get('status') == 'frozen' and protocol.get('no_test_or_validation_tuning') is True,
          '事前設定尚未凍結，或沒有禁止測試集調參的紀錄')
    check(protocol.get('test_csv_sha256') == csv_sha, 'Protocol 與原始 caption CSV 不一致')
    parsed = urlparse(args.artifact_url or '')
    check(parsed.scheme == 'https' and bool(parsed.netloc)
          and parsed.hostname not in {'localhost', '127.0.0.1', '::1'}, '缺少有效的公開成果資料夾 HTTPS 連結')
    identity = gen.get('identity', {})
    check(gen.get('complete') is True and gen.get('formal_complete') is True
          and identity.get('mode') == 'formal_test', '100 首正式生成尚未完成')
    samples = gen.get('samples', [])
    records = {r.get('id'): r for r in samples}
    check(len(samples) == 100 and set(records) == expected, 'Generation manifest 沒有完整且唯一的 100 個 ID')
    check(identity.get('csv_sha256') == csv_sha, '生成使用的 caption 已改變')
    setting = identity.get('settings', {})
    check(setting.get('steps') == 100 and setting.get('seconds') == 10.0
          and setting.get('guidance_scale') == 7.0 and setting.get('schedule') == 'constant'
          and setting.get('rescale') == 0.0 and setting.get('device') in {'mps', 'cpu', 'cuda'}
          and setting.get('dtype') in {'float16', 'float32'} and bool(setting.get('mp3_bitrate')),
          '生成設定不符合這份固定 Baseline 報告')
    if protocol:
        check(identity.get('protocol', {}).get('sha256') == sha(args.protocol), '生成與本份 frozen protocol 不一致')
        main = protocol.get('main_submission_variant')
        check(main == 'baseline', '本份報告固定描述事前選定的 Baseline 提交')
        check(identity.get('protocol', {}).get('variant') == main, '生成不是事先選定的主提交方法')
        conditions = {c.get('id'): c.get('settings') for c in protocol.get('generation_conditions', [])}
        check(identity.get('settings') == conditions.get(main), '主提交設定與 protocol 不一致')
        for name in LABELS:
            actual = conditions.get(name) or {}
            scheduled, rescaled = name in {'schedule_only', 'combined'}, name in {'rescale_only', 'combined'}
            check(actual.get('guidance_scale') == (10.0 if scheduled else 7.0)
                  and actual.get('schedule') == ('linear' if scheduled else 'constant')
                  and actual.get('rescale') == (0.5 if rescaled else 0.0)
                  and (not scheduled or actual.get('end_scale') == 4.0),
                  '額外方法與報告的同均值 7／r=0.5 設計不一致：' + name)
    for i, row in records.items():
        check(i in data['captions'] and row.get('caption_sha256') == text_sha(data['captions'].get(i, '')),
              '生成 caption 雜湊不一致：' + str(i))
        audit = row.get('audio_audit', {})
        check(finite(row.get('generation_seconds')) and row['generation_seconds'] > 0
              and type(row.get('tokenization', {}).get('truncated')) is bool
              and all(finite(audit.get(k)) for k in ('rms', 'peak', 'fraction_abs_ge_1')),
              '生成耗時、token 或訊號稽核資料缺漏：' + str(i))
        scheduler = row.get('sampling', {}).get('scheduler', {})
        scheduler_config = scheduler.get('config', {})
        check(scheduler.get('class', '').split('.')[-1] == 'CosineDPMSolverMultistepScheduler'
              and scheduler_config.get('solver_order') == 2
              and scheduler_config.get('solver_type') == 'midpoint'
              and scheduler_config.get('prediction_type') == 'v_prediction'
              and scheduler_config.get('sigma_schedule') == 'exponential',
              '缺少報告所描述的實際 diffusion scheduler 紀錄：' + str(i))
        compatibility = scheduler.get('compatibility', {})
        check(compatibility.get('applied') is True
              and compatibility.get('id') == 'cosine_zero_sigma_noise_guard_v1'
              and all(compatibility.get(k) is True for k in
                      ('terminal_only', 'stock_class_and_config_preserved',
                       'nonterminal_steps_unmodified', 'native_comparison_uses_same_guard')),
              '缺少自訂與官方對照共用末端零噪聲修正的執行紀錄：' + str(i))
        rel = Path(row.get('mp3', {}).get('path', ''))
        if rel.is_absolute() or '..' in rel.parts or not str(rel).endswith('.mp3'):
            reasons.append('不合法的生成 MP3 路徑：' + str(i))
            continue
        path = args.generation_dir / rel
        check(path.is_file() and not path.is_symlink() and sha(path) == row.get('mp3', {}).get('sha256'),
              '報告引用的 MP3 已改變或缺少：' + str(i))
    check(meta.get('status') == 'retrieval_complete' and meta.get('reference_count') == 5367
          and meta.get('caption_count') == 100, '檢索沒有完成全部 100 × 5367 筆比較')
    check(meta.get('model', {}).get('checkpoint_sha256') == CLAP_SHA, '檢索使用了不同的 CLAP 權重')
    check(set(retrieval) == expected and all(isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_-]{11}', v)
                                           for v in retrieval.values()), 'retrieval.json 格式或 ID 不完整')
    check(meta.get('unique_retrieved_references') == len(set(retrieval.values())), '不同檢索 reference 的數量不一致')
    if paths['retrieval'].is_file():
        check(meta.get('retrieval_json_sha256') == sha(paths['retrieval']), '檢索 JSON 與執行紀錄不一致')
    check(meta.get('caption_source_sha256') == csv_sha, '檢索使用的原始 captions 不一致')
    if paths['protocol'].is_file():
        check(meta.get('protocol_sha256') == sha(paths['protocol']), '檢索與生成不是同一份 frozen protocol')
    check(set(details) == expected, '缺少檢索排名及 token 截斷紀錄')
    for i, row in details.items():
        check(row.get('original_caption') == data['captions'].get(i), '檢索 caption 被改寫：' + str(i))
        ranked = row.get('topk', [])
        check(bool(ranked) and ranked[0].get('musiccaps_id') == retrieval.get(i), 'Top-1 與提交 ID 不一致：' + str(i))
        check(len(ranked) >= 2 and all(finite(r.get('cosine')) for r in ranked[:2])
              and type(row.get('tokenizer', {}).get('was_truncated')) is bool,
              '缺少真實 cosine margin 或 tokenizer 紀錄：' + str(i))
    check(meta.get('captions_truncated') == sum(r.get('tokenizer', {}).get('was_truncated') is True for r in details.values()),
          'CLAP 截斷總數與逐筆紀錄不一致')
    check(evaluation.get('completed') is True, '必要的完整 100 首評估尚未完成')
    counts = evaluation.get('counts', {})
    check(counts.get('generated_clips') == 100 and counts.get('fd_reference_clips') == 5367
          and counts.get('paired_reference_rows') == 100, '完整評估的樣本數不符合規定')
    for key in ('FD-VGG-all-reference', 'FD-VGG-retrieved-proxy', KL, 'CLAP'):
        check(finite(evaluation.get(key)), '必要指標無有效數值：' + key)
    check(evaluation.get('reference_is_ground_truth') is False, 'Retrieved reference 必須明確標成 proxy')
    check(evaluation.get('caption_csv_sha256') == csv_sha, '評估 caption CSV 不一致')
    if paths['retrieval'].is_file():
        check(evaluation.get('retrieval_sha256') == sha(paths['retrieval']), '評估配對與檢索 JSON 不一致')
    check(evaluation.get('clap_provenance', {}).get('checkpoint_sha256') == CLAP_SHA,
          '評估未使用指定的 CLAP 權重')
    for block, keys in [('clap_pairing_control', ('matched_mean', 'cyclic_mismatch_mean',
                                                 'matched_minus_mismatch_mean', 'matched_greater_fraction')),
                        ('clap_silence_control', ('silence_mean', 'generated_minus_silence_mean',
                                                 'generated_greater_fraction'))]:
        check(all(finite(evaluation.get(block, {}).get(k)) for k in keys), '缺少真正的 CLAP 錯配／靜音控制結果：' + block)
    scored_inputs = {r.get('id'): r.get('source_sha256') for r in evaluation.get('clap_audio_inputs', [])}
    check(scored_inputs == {i: r.get('mp3', {}).get('sha256') for i, r in records.items()} and len(scored_inputs) == 100,
          '評估輸入不是這一批實際提交的 MP3')
    per_score = {r.get('id'): r for r in evaluation.get('per_caption', [])}
    check(set(per_score) == expected and len(evaluation.get('per_caption', [])) == 100,
          '缺少完整逐首 CLAP／KL 評估')
    for i, row in per_score.items():
        check(finite(row.get('clap')) and finite(row.get('kl_passt_softmax'))
              and row.get('reference_id') == retrieval.get(i), '逐首分數或 reference 配對不完整：' + str(i))
    if len(per_score) == 100 and all(finite(r.get('clap')) and finite(r.get('kl_passt_softmax')) for r in per_score.values()):
        check(abs(statistics.mean(r['clap'] for r in per_score.values()) - evaluation.get('CLAP', math.inf)) < 1e-5,
              '完整 CLAP 平均與逐首分數不一致')
        check(abs(statistics.mean(r['kl_passt_softmax'] for r in per_score.values()) - evaluation.get(KL, math.inf)) < 1e-5,
              '完整 KL 平均與逐首分數不一致')
    keyed = {}
    if not getattr(args, 'baseline_only', False):
        keyed = validate_ablation(args, protocol, ablation, paths, csv_sha, check, reasons)
        validate_analysis_statistics(ablation, keyed, check)
    parity = protocol.get('probe_evidence', {}).get('actual_pretrained_waveform_parity', {})
    check(parity.get('passed') is True and finite(parity.get('max_abs_difference'))
          and finite(parity.get('absolute_tolerance')) and parity['absolute_tolerance'] >= 0
          and parity['max_abs_difference'] <= parity['absolute_tolerance'],
          '缺少真正預訓練模型與原生 pipeline 的 waveform parity 證據')
    data.update(records=records, per_score=per_score, ablation_rows=keyed,
                issues=list(dict.fromkeys(reasons)), paths=paths)
    if not args.draft and data['issues']:
        raise ValueError('正式報告尚不能建立：\n- ' + '\n- '.join(data['issues']))
    return data


def validate_ablation(args, protocol, ablation, paths, csv_sha, check, reasons):
    check(ablation.get('completed') is True and ablation.get('not_the_required_full_100_clip_evaluation') is True,
          '固定子集的額外實驗尚未完成')
    cohort = protocol.get('ablation', {}).get('cohort_ids', [])
    factor_ids = [r.get('id') for r in protocol.get('ablation', {}).get('conditions', [])]
    check(len(cohort) == 20 and ablation.get('cohort_ids') == cohort and ablation.get('n_captions') == 20
          and ablation.get('n_conditions') == 4, '額外實驗不是凍結的 20 筆四條件設計')
    check(cohort == [str(i) for i in range(20)] and set(factor_ids) == set(LABELS),
          '報告固定描述 Baseline／排程／幅度修正／Combined 與原始 ID 0–19')
    check(set(ablation.get('summary', {})) == set(factor_ids) and len(factor_ids) == 4,
          '額外實驗缺少凍結的條件')
    if protocol:
        check(ablation.get('condition_settings') == {c['id']: c['settings'] for c in protocol.get('generation_conditions', [])},
              '額外實驗的實際設定與 frozen protocol 不一致')
    for metric in ('CLAP', KL, 'rms', 'peak', 'fraction_abs_ge_1'):
        estimates = [ablation.get('summary', {}).get(name, {}).get(metric, {}) for name in factor_ids]
        estimates += [ablation.get('paired_deltas_vs_baseline', {}).get(metric, {}).get(name, {})
                      for name in factor_ids if name != 'baseline']
        estimates += [ablation.get('factorial_effects', {}).get(metric, {}).get(name, {}) for name in
                      ('schedule_main_effect', 'rescale_main_effect', 'interaction_difference_in_differences')]
        for estimate in estimates:
            bounds = estimate.get('percentile_95_ci', [])
            check(finite(estimate.get('mean')) and len(bounds) == 2 and all(finite(v) for v in bounds)
                  and bounds[0] <= bounds[1] and estimate.get('n_captions') == 20,
                  '額外實驗缺少有效的配對估計或區間：' + metric)
    if paths['protocol'].is_file():
        check(ablation.get('provenance', {}).get('protocol_sha256') == sha(paths['protocol']),
              '額外實驗與 frozen protocol 不一致')
    check(ablation.get('provenance', {}).get('captions_sha256') == csv_sha, '額外實驗 caption CSV 不一致')
    if paths['retrieval'].is_file():
        check(ablation.get('provenance', {}).get('retrieval_sha256') == sha(paths['retrieval']),
              '額外實驗使用了不同的 reference 配對')
    ablation_rows = ablation.get('per_caption', [])
    keyed = {(r.get('condition'), r.get('id')): r for r in ablation_rows}
    check(len(keyed) == len(ablation_rows) == 80 and set(keyed) == {(c, i) for c in factor_ids for i in cohort},
          '額外實驗沒有完整 80 個條件 × caption 紀錄')
    csv_path = args.ablation.with_name('ablation_per_caption.csv')
    if csv_path.is_file():
        with csv_path.open(newline='') as handle:
            csv_rows = list(csv.DictReader(handle))
        csv_keys = [(r.get('condition'), r.get('id')) for r in csv_rows]
        check(len(csv_rows) == len(set(csv_keys)) == 80 and set(csv_keys) == set(keyed),
              '額外實驗 CSV 列數、唯一性或覆蓋不正確')
        for row in csv_rows:
            reference = keyed.get((row.get('condition'), row.get('id')), {})
            for key in ('CLAP', KL, 'rms', 'peak', 'fraction_abs_ge_1', 'CLAP_delta_vs_baseline', KL + '_delta_vs_baseline'):
                try:
                    check(finite(reference.get(key)) and abs(float(row[key]) - reference[key]) < 1e-8,
                          '額外實驗 CSV／JSON 不一致：' + key)
                except (KeyError, TypeError, ValueError):
                    reasons.append('額外實驗 CSV 無有效數值：' + key)
    else:
        reasons.append('缺少 ablation_per_caption.csv')
    return keyed


def validate_analysis_statistics(ablation, keyed, check):
    """Reconcile report claims with the actual paired rows, without new tuning."""
    for name in LABELS:
        fd = ablation.get('summary', {}).get(name, {}).get('FD-VGG-cohort-vs-retrieved-proxy')
        check(finite(fd) and fd >= -1e-6, '缺少同批 20 題的真實 FD：' + name)
    wanted = {(name, str(i)) for name in LABELS for i in range(20)}
    if set(keyed) != wanted:
        return  # The original validation already rejects incomplete formal data.
    for metric in ('CLAP', KL, 'rms', 'peak', 'fraction_abs_ge_1'):
        if not all(finite(row.get(metric)) for row in keyed.values()):
            check(False, '額外實驗逐題數值缺漏：' + metric)
            continue
        values = {name: [keyed[name, str(i)][metric] for i in range(20)] for name in LABELS}
        for name in LABELS:
            summary = ablation.get('summary', {}).get(name, {}).get(metric, {}).get('mean')
            check(finite(summary) and abs(summary - statistics.mean(values[name])) < 1e-5,
                  '條件平均與逐題資料不一致：' + name + '/' + metric)
            if name != 'baseline':
                delta = [a - b for a, b in zip(values[name], values['baseline'])]
                reported = ablation.get('paired_deltas_vs_baseline', {}).get(metric, {}).get(name, {}).get('mean')
                check(finite(reported) and abs(reported - statistics.mean(delta)) < 1e-5,
                      '配對差平均與逐題資料不一致：' + name + '/' + metric)
        a, b, c, d = [values[n] for n in LABELS]
        effects = {
            'schedule_main_effect': [(bi - ai + di - ci) / 2 for ai, bi, ci, di in zip(a, b, c, d)],
            'rescale_main_effect': [(ci - ai + di - bi) / 2 for ai, bi, ci, di in zip(a, b, c, d)],
            'interaction_difference_in_differences': [di - bi - ci + ai for ai, bi, ci, di in zip(a, b, c, d)]}
        for name, values in effects.items():
            reported = ablation.get('factorial_effects', {}).get(metric, {}).get(name, {}).get('mean')
            check(finite(reported) and abs(reported - statistics.mean(values)) < 1e-5,
                  '主效應／交互作用與逐題資料不一致：' + name + '/' + metric)


def paired_counts(data, name):
    rows = [r for (condition, _), r in data['ablation_rows'].items() if condition == name]
    if len(rows) != 20 or not all(finite(r.get('CLAP_delta_vs_baseline'))
                                and finite(r.get(KL + '_delta_vs_baseline')) for r in rows):
        return None
    return {'n': len(rows),
            'clap_up': sum(r['CLAP_delta_vs_baseline'] > 0 for r in rows),
            'clap_down': sum(r['CLAP_delta_vs_baseline'] < 0 for r in rows),
            'both_better': sum(r['CLAP_delta_vs_baseline'] > 0 and r[KL + '_delta_vs_baseline'] < 0 for r in rows),
            'both_worse': sum(r['CLAP_delta_vs_baseline'] < 0 and r[KL + '_delta_vs_baseline'] > 0 for r in rows),
            'tradeoff': sum(r['CLAP_delta_vs_baseline'] * r[KL + '_delta_vs_baseline'] > 0 for r in rows)}


def control_diagnostics(data):
    rows = list(data['per_score'].values())
    valid = [r for r in rows if all(finite(r.get(k)) for k in
                                   ('clap', 'clap_cyclic_mismatch', 'clap_silence'))]
    if len(valid) != 100:
        return None
    return {'n': 100,
            'mismatch_not_better_ids': [r['id'] for r in valid if r['clap'] <= r['clap_cyclic_mismatch']],
            'silence_not_better_ids': [r['id'] for r in valid if r['clap'] <= r['clap_silence']]}


def estimate_reading(estimate, higher_better=True):
    bounds = estimate.get('percentile_95_ci', []) if isinstance(estimate, dict) else []
    if len(bounds) != 2 or not all(finite(v) for v in bounds):
        return '待完成'
    if bounds[0] <= 0 <= bounds[1]:
        return '區間含 0，未確認平均改善'
    better = bounds[0] > 0 if higher_better else bounds[1] < 0
    return '區間支持改善方向' if better else '區間支持變差方向'


def fd_discussion(ablation):
    summary = ablation.get('summary', {})
    key = 'FD-VGG-cohort-vs-retrieved-proxy'
    values = {name: summary.get(name, {}).get(key) for name in LABELS}
    if not all(finite(v) for v in values.values()):
        return '同批 20 題的 FD 與逐題指標是否一致，待完整結果後判讀。'
    changes = {name: values[name] - values['baseline'] for name in LABELS if name != 'baseline'}
    line = '同批 ΔFD：' + '、'.join(f'{LABELS[name]} {change:+.3f}' for name, change in changes.items()) + '。'
    for name in ('rescale_only', 'combined', 'schedule_only'):
        dk = ablation.get('paired_deltas_vs_baseline', {}).get(KL, {}).get(name, {}).get('mean')
        if finite(dk) and dk < 0 and changes[name] > 0:
            return line + f'{LABELS[name]}的平均 KL 較低、FD 卻較高，逐題類別接近度與整批特徵分布呈現取捨；20 題 FD 未估計區間。'
    return line + 'FD 越低越接近同一 proxy 分布；這是小樣本描述值，未估計區間，不能據此宣稱穩定優勢。'


def paired_tradeoff_figure(data, args):
    """Export a standard scientific plot; points are actual paired captions."""
    rows = [r for (name, _), r in data['ablation_rows'].items() if name == 'combined']
    if len(rows) != 20 or not hasattr(args, 'figure_dir'):
        return None
    # Keep font/config caches local to this assignment, including offline runs.
    for variable, directory in [('MPLCONFIGDIR', ROOT / 'cache/report_matplotlib'),
                                 ('XDG_CACHE_HOME', ROOT / 'cache/report_fonts')]:
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(directory))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=str(args.font))
    rows.sort(key=lambda r: int(r['id']))
    xs = [r['CLAP_delta_vs_baseline'] for r in rows]
    ys = [-r[KL + '_delta_vs_baseline'] for r in rows]
    if not all(finite(v) for v in xs + ys):
        raise ValueError('Cannot plot incomplete paired values')
    fig, axis = plt.subplots(figsize=(6.2, 2.55), layout='constrained')
    axis.axhline(0, color='.6', linewidth=.75, linestyle='--')
    axis.axvline(0, color='.6', linewidth=.75, linestyle='--')
    axis.scatter(xs, ys, s=25, color='.15', alpha=.8)
    for i in {min(range(20), key=lambda i: xs[i]), max(range(20), key=lambda i: xs[i])}:
        axis.annotate('ID ' + rows[i]['id'], (xs[i], ys[i]), xytext=(5, 5),
                      textcoords='offset points', fontsize=9)
    axis.set_xlabel('ΔCLAP ↑', fontproperties=font, fontsize=11)
    axis.set_ylabel('−ΔKL proxy ↑', fontproperties=font, fontsize=11)
    axis.set_title('Combined 每題變化（n=20）' + ('：草稿' if args.draft else ''),
                   fontproperties=font, fontsize=11)
    axis.text(.99, .97, '右上：兩指標皆改善', transform=axis.transAxes,
              ha='right', va='top', fontproperties=font, fontsize=9, color='.35')
    axis.tick_params(labelsize=9)
    axis.spines[['top', 'right']].set_visible(False)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    stem = 'combined_tradeoff' + ('_DRAFT' if args.draft else '')
    png, pdf = [args.figure_dir / (stem + suffix) for suffix in ('.png', '.pdf')]
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    data['paths'].update(tradeoff_figure_png=png, tradeoff_figure_pdf=pdf)
    return png


def analysis_discussion(data):
    ablation = data['ablation']
    deltas = ablation.get('paired_deltas_vs_baseline', {})
    lines = []
    for name in ('schedule_only', 'rescale_only', 'combined'):
        dc = deltas.get('CLAP', {}).get(name, {})
        dk = deltas.get(KL, {}).get(name, {})
        lines.append(f"{LABELS[name]}：ΔCLAP {number(dc.get('mean'))}，{estimate_reading(dc)}；"
                     f"ΔKL {number(dk.get('mean'))}，{estimate_reading(dk, False)}。")
    lines[0] += '本排程早期較強，與 [7] 圖像實驗偏好中段 guidance 不同；不能把該文當成本音訊排程有效的證據。'
    counts = paired_counts(data, 'combined')
    if counts:
        lines.append(f"Combined 逐題檢查：{counts['both_better']}/20 題兩指標同時改善，"
                     f"{counts['both_worse']}/20 題同時下降，{counts['tradeoff']}/20 題呈現取捨；"
                     '其餘為至少一指標持平。平均差不能取代逐題檢查。')
    else:
        lines.append('Combined 逐題改善／下降與指標取捨，待完整配對結果後填入。')
    lines.append(fd_discussion(ablation))
    interaction = ablation.get('factorial_effects', {}).get('CLAP', {}).get('interaction_difference_in_differences', {})
    bounds = interaction.get('percentile_95_ci', [])
    if len(bounds) != 2 or not all(finite(v) for v in bounds):
        mechanism = '兩項改動是否互相影響，待交互作用區間完成後判讀。'
    elif bounds[0] > 0:
        mechanism = 'CLAP 交互作用區間全為正，合用的變化大於兩項各自變化的加總。'
    elif bounds[1] < 0:
        mechanism = 'CLAP 交互作用區間全為負，合用的變化小於兩項各自變化的加總。'
    else:
        mechanism = 'CLAP 交互作用區間含 0，目前無法確認兩項改動的效果超出加總。'
    lines.append(mechanism + '幅度修正作用在每步 v 預測，不是 MP3 音量正規化；RMS／peak 變化不能證明音質改善。')
    lines.append('限制與下一步：前 20 題非隨機、每題一個 seed，區間未校正多重比較。'
                 '後續以獨立開發題做多 seed 重複及盲聽，確認效果是否跨種子成立；不據此回頭調 test 或挑提交音訊。'
                 '本次未做人類偏好測試，MQAScore 選做未實作。')
    return lines


def discussion_blocks(data):
    """Lead with concrete findings, then explain their scope and tradeoffs."""
    a = data['ablation']
    if a.get('completed') is not True:
        return None
    summary, deltas = a['summary'], a['paired_deltas_vs_baseline']
    base, rescale = summary['baseline'], summary['rescale_only']
    dc = deltas['CLAP']['rescale_only']; rms = deltas['rms']['rescale_only']
    counts = paired_counts(data, 'rescale_only'); combined = paired_counts(data, 'combined')
    uncertain = all(v['percentile_95_ci'][0] <= 0 <= v['percentile_95_ci'][1]
                    for metric in ('CLAP', KL) for v in deltas[metric].values())
    evidence = ('三種改動的 CLAP、KL 配對區間都含 0，尚未確認平均改善。' if uncertain else
                '各指標的區間方向不同，需分別解讀，不能由單一點估計概括。')
    interaction = a['factorial_effects']['CLAP']['interaction_difference_in_differences']
    ib = interaction['percentile_95_ci']
    interaction_note = ('區間含 0，沒有確認疊加效果，也不能斷言兩項改動互相抵消。' if ib[0] <= 0 <= ib[1] else
                        '區間排除 0，支持此子集有交互作用；不能推論所有 captions 或 seeds。')
    return [
        ('平均略升，但改善仍不確定',
         f"只做幅度修正：CLAP {base['CLAP']['mean']:.4f} → {rescale['CLAP']['mean']:.4f}，"
         f"{counts['clap_up']}/20 題上升、{counts['clap_down']}/20 題下降；Δ {interval(dc)}。" + evidence),
        ('RMS 的變化有區間支持，不能直接當成音質改善',
         f"幅度修正後，平均 RMS {base['rms']['mean']:.4f} → {rescale['rms']['mean']:.4f}；"
         f"配對 Δ {interval(rms)}。FD proxy 卻由 {base['FD-VGG-cohort-vs-retrieved-proxy']:.3f} "
         f"變成 {rescale['FD-VGG-cohort-vs-retrieved-proxy']:.3f}。"
         'prediction rescale 改變了訊號，但幅度下降不等於文字對齊或音樂性變好。'),
        ('兩者合用仍有失敗，平均值不能取代逐題檢查',
         f"Combined 有 {combined['both_better']}/20 題兩指標同時改善、{combined['both_worse']}/20 題同時下降、"
         f"{combined['tradeoff']}/20 題取捨。CLAP 交互作用 {interval(interaction)}；" + interaction_note),
        ('限制與下一步',
         '前 20 題不是隨機抽樣，每題只有一個 seed，區間未校正多重比較；未做人類偏好測試，MQAScore 選做未實作。'
         '後續應在獨立開發資料預先設定多 seed 比較與盲聽，確認效果是否可重現；不回頭調這批 test 或挑提交音訊。')]


class Slides:
    def __init__(self, path, draft, total_pages=10):
        self.canvas = Canvas(str(path), pagesize=(WIDTH, HEIGHT), pageCompression=1)
        title = 'HW3：文字到音樂 Baseline 成果' if total_pages == 7 else 'HW3：文字到音樂與 CFG 對照實驗'
        self.canvas.setTitle(title + ' — ' + STUDENT)
        self.canvas.setAuthor(STUDENT)
        self.draft = draft
        self.total_pages = total_pages
        self.page = 0

    def para(self, text, x=LEFT, y=510, width=RIGHT - LEFT, size=16, leading=None, limit=None, gray=False):
        style = ParagraphStyle('body', fontName=FONT, fontSize=size, leading=leading or size * 1.42,
                               textColor=colors.HexColor('#586777') if gray else INK if size >= 23 else BODY,
                               wordWrap='CJK', allowWidows=0, allowOrphans=0)
        markup = escape(str(text)).replace('\n', '<br/>')
        p = Paragraph(markup, style)
        _, height = p.wrap(width, HEIGHT)
        if y - height < (limit or CONTENT_BOTTOM):
            raise ValueError(f'Report page {self.page} overflows; shorten the content instead of hiding it')
        p.drawOn(self.canvas, x, y - height)
        return y - height

    def callout(self, text, y, size=16):
        style = ParagraphStyle('callout', fontName=FONT, fontSize=size, leading=size * 1.4,
                               textColor=INK, wordWrap='CJK')
        p = Paragraph(escape(str(text)), style)
        h = p.wrap(RIGHT - LEFT - 36, HEIGHT)[1] + 24
        if y - h < CONTENT_BOTTOM:
            raise ValueError('Key finding callout would overflow')
        self.canvas.setFillColor(PALE)
        self.canvas.rect(LEFT, y - h, RIGHT - LEFT, h, fill=1, stroke=0)
        self.canvas.setFillColor(INK)
        self.canvas.rect(LEFT, y - h, 3, h, fill=1, stroke=0)
        p.drawOn(self.canvas, LEFT + 18, y - h + 12)
        return y - h

    def cards(self, cards, y=360, height=105):
        gap = 18
        width = (RIGHT - LEFT - gap * (len(cards) - 1)) / len(cards)
        for i, (label, value, note) in enumerate(cards):
            x = LEFT + i * (width + gap)
            self.canvas.setFillColor(PALE)
            self.canvas.rect(x, y - height, width, height, fill=1, stroke=0)
            self.para(label, x=x + 16, y=y - 12, width=width - 32, size=12, gray=True)
            self.para(value, x=x + 16, y=y - 37, width=width - 32, size=22)
            self.para(note, x=x + 16, y=y - 77, width=width - 32, size=11.5, gray=True)

    def bullets(self, texts, y=505, size=16, gap=15, width=None):
        for text in texts:
            self.canvas.setFillColor(colors.black)
            self.canvas.circle(LEFT + 3, y - size * .55, 2, fill=1, stroke=0)
            y = self.para(text, LEFT + 18, y, width or RIGHT - LEFT - 18, size) - gap
        return y

    def table(self, headers, widths, rows, y=500, size=14):
        if abs(sum(widths) - (RIGHT - LEFT)) > .01:
            raise ValueError('Table widths must match the content area')
        c = self.canvas
        style = ParagraphStyle('cell', fontName=FONT, fontSize=size, leading=size * 1.35,
                               wordWrap='CJK', textColor=BODY)
        header = ParagraphStyle('header', parent=style, textColor=INK)
        for index, row in enumerate([headers] + rows):
            paragraphs, heights = [], []
            for value, width in zip(row, widths):
                p = Paragraph(escape(str(value)).replace('\n', '<br/>'), header if index == 0 else style)
                heights.append(p.wrap(width - 18, HEIGHT)[1]); paragraphs.append(p)
            height = max(heights) + 22
            if y - height < CONTENT_BOTTOM:
                raise ValueError(f'Table on page {self.page} overflows')
            if index == 0:
                c.setFillColor(PALE); c.rect(LEFT, y - height, sum(widths), height, fill=1, stroke=0)
            elif index % 2 == 0:
                c.setFillColor(colors.HexColor('#fafbfd')); c.rect(LEFT, y - height, sum(widths), height, fill=1, stroke=0)
            x = LEFT
            for p, width, h in zip(paragraphs, widths, heights):
                p.drawOn(c, x + 9, y - 11 - h); x += width
            c.setStrokeColor(colors.HexColor('#cbd5df')); c.setLineWidth(.45)
            c.line(LEFT, y - height, RIGHT, y - height)
            y -= height
        return y

    def start(self, title, subtitle=None):
        if self.page:
            self.canvas.showPage()
        self.page += 1
        c = self.canvas
        c.setFillColor(colors.white); c.rect(0, 0, WIDTH, HEIGHT, fill=1, stroke=0)
        self.para(title, y=HEIGHT - 40, size=26)
        c.setStrokeColor(colors.HexColor('#cbd5df')); c.setLineWidth(.6)
        c.line(LEFT, HEIGHT - 90, RIGHT, HEIGHT - 90)
        c.setFillColor(INK); c.rect(LEFT, HEIGHT - 92, 66, 3, fill=1, stroke=0)
        if subtitle:
            self.para(subtitle, y=HEIGHT - 104, size=12, gray=True)
        c.setStrokeColor(colors.HexColor('#dde3e9')); c.line(LEFT, 43, RIGHT, 43)
        c.setFillColor(colors.HexColor('#586777')); c.setFont(FONT, 10)
        c.drawString(LEFT, 27, 'HW3  |  ' + STUDENT)
        c.drawRightString(RIGHT, 27, f'{self.page:02d} / {self.total_pages:02d}')
        if self.draft:
            c.setFont(FONT, 11); c.drawCentredString(WIDTH / 2, 25, '草稿：非正式提交版本')

    def save(self):
        if self.page != self.total_pages:
            raise ValueError(f'The selected report scope must have exactly {self.total_pages} pages')
        self.canvas.save()


def samples_summary(data):
    times = [r.get('generation_seconds') for r in data['records'].values() if finite(r.get('generation_seconds'))]
    audits = [r.get('audio_audit', {}) for r in data['records'].values()]
    return {'time_median': statistics.median(times) if times else None,
            'time_total': sum(times) if times else None,
            'clipped': sum(finite(a.get('fraction_abs_ge_1')) and a['fraction_abs_ge_1'] > 0 for a in audits),
            't5_truncated': sum(r.get('tokenization', {}).get('truncated') is True for r in data['records'].values())}


def select_cases(data):
    candidates = [r for (condition, _), r in data['ablation_rows'].items()
                  if condition == 'combined' and finite(r.get('CLAP_delta_vs_baseline'))]
    if not candidates:
        return None, None
    ordered = sorted(candidates, key=lambda r: (r['CLAP_delta_vs_baseline'], int(r['id'])))
    return ordered[-1], ordered[0]


def select_baseline_cases(data):
    rows = sorted((r for r in data['per_score'].values() if finite(r.get('clap'))),
                  key=lambda r: (r['clap'], int(r['id'])))
    return (rows[0], rows[(len(rows) - 1) // 2]) if rows else (None, None)


def scheduler_description(data):
    """Prefer recorded execution; only drafts may display the local model config."""
    recorded = [r.get('sampling', {}).get('scheduler', {}) for r in data['records'].values()]
    if recorded:
        signatures = {json.dumps(s, sort_keys=True) for s in recorded}
        if len(signatures) != 1:
            # Global RNG seeds legitimately vary by caption; compare the actual
            # scheduler class/config, not the seed-bearing audit wrapper.
            signatures = {json.dumps({'class': s.get('class'), 'config': s.get('config')}, sort_keys=True)
                          for s in recorded}
        if len(signatures) != 1:
            raise ValueError('Actual diffusion scheduler class/config differs across captions')
        scheduler = recorded[0]
        name, config = scheduler.get('class', '').split('.')[-1], scheduler.get('config', {})
        label = 'Diffusion scheduler（實際）'
    else:
        # This is configuration evidence, never a completed pretrained run.
        path = ROOT / 'cache/models/stable-audio-open-1.0-f21265c1/scheduler/scheduler_config.json'
        config = json.loads(path.read_text()) if path.is_file() else {}
        name = config.get('_class_name', '待完成')
        label = 'Diffusion scheduler（草稿設定）'
    details = (f"{name}\norder={config.get('solver_order', '待完成')} / {config.get('solver_type', '待完成')} / "
               f"{config.get('prediction_type', '待完成')} / {config.get('sigma_schedule', '待完成')} sigma")
    return label, details


def compatibility_disclosure(data):
    records = list(data['records'].values())
    if not records:
        return '末端零噪聲修正仍待真實 pilot 驗證；官方 pipeline 對照亦將套用相同修正，並非完全未修改的上游執行。'
    guards = [r.get('sampling', {}).get('scheduler', {}).get('compatibility', {}) for r in records]
    if not all(g.get('applied') is True and g.get('native_comparison_uses_same_guard') is True for g in guards):
        return '末端零噪聲修正的共用紀錄尚未完整；目前不可聲稱已完成同條件的官方 pipeline 對照。'
    return '官方 pipeline 對照亦套用相同末端零噪聲修正：僅 σ_next=0 時用零噪聲，因其係數原為0；前99步不變。這是相容修正，並非完全未修改的上游執行。'


def combined_conclusion(ablation):
    delta = ablation.get('paired_deltas_vs_baseline', {})
    clap = delta.get('CLAP', {}).get('combined', {})
    kl = delta.get(KL, {}).get('combined', {})
    bounds = clap.get('percentile_95_ci', [])
    if len(bounds) != 2 or not all(finite(v) for v in bounds) or not finite(kl.get('mean')):
        return 'Combined 的實際取捨要等配對結果完成後再判讀；目前沒有可支持改善的真實實驗結論。'
    if bounds[0] > 0:
        finding = 'Combined 在這 20 題的 CLAP 差值區間全為正，顯示這組樣本的文字對齊較強。'
    elif bounds[1] < 0:
        finding = 'Combined 在這 20 題的 CLAP 差值區間全為負，這組樣本沒有支持原先的改善假設。'
    else:
        finding = 'Combined 的 CLAP 差值區間包含 0，這組樣本不足以支持一致改善。'
    if kl['mean'] == 0:
        return finding + 'KL 平均差為 0，與 Baseline 相同；這不是人類品質判斷。'
    direction = '更接近' if kl['mean'] < 0 else '較遠離'
    return finding + f"KL 平均差 {number(kl['mean'])}，表示{direction}檢索 proxy；這不是人類品質判斷。"


def render(slides, data, args):
    baseline_only = getattr(args, 'baseline_only', False)
    protocol, gen, evaluation, ablation = [data[k] for k in ('protocol', 'generation', 'evaluation', 'ablation')]
    meta, details = data['retrieval_manifest'], data['retrieval_details']
    settings = gen.get('identity', {}).get('settings', {})
    summary = samples_summary(data)
    parity = protocol.get('probe_evidence', {}).get('actual_pretrained_waveform_parity', {})
    conditions = protocol.get('generation_conditions', [])
    conditions = conditions or [{'id': k, 'settings': {}} for k in LABELS]
    cohort = protocol.get('ablation', {}).get('cohort_ids', [])
    scheduler_label, scheduler_details = scheduler_description(data)

    slides.start('HW3：文字到音樂 Baseline 成果' if baseline_only else '文字到音樂：CFG 排程與幅度修正',
                 'HW3  /  Stable Audio Open 1.0  /  r14942154')
    if baseline_only:
        slides.para('完成 100 筆檢索、CFG、生成與必要評估。', y=455, size=24)
        slides.bullets(['100 個原始 captions：檢索、生成與三種必要評估。',
                        '本版僅呈現已驗證的 Baseline，不列未完成的額外實驗。',
                        '方法與種子事先固定；指標不當作人類品質評分。'], y=355, size=17, gap=18)
    else:
        dc = ablation.get('paired_deltas_vs_baseline', {}).get('CLAP', {})
        ready = all(finite(dc.get(name, {}).get('mean')) for name in LABELS if name != 'baseline')
        title = '平均分數略升，改善仍待確認' if ready and all(
            dc[name]['mean'] > 0 and dc[name]['percentile_95_ci'][0] <= 0 <= dc[name]['percentile_95_ci'][1]
            for name in LABELS if name != 'baseline') else '用四條件對照，檢查改動是否有效'
        slides.para(title, y=480, size=30)
        slides.para('先完成 100 首主提交，再固定 20 題，比較排程、幅度修正與兩者合用。', y=425, size=18)
        slides.cards([
            ('完整 100 首 Baseline', 'CLAP ' + number(evaluation.get('CLAP')),
             'FD ' + number(evaluation.get('FD-VGG-all-reference'), 3) + '  /  KL ' + number(evaluation.get(KL))),
            ('額外配對實驗', '2 × 2 四條件', '相同 20 題、初始雜訊、模型與步數'),
            ('只做幅度修正', 'ΔCLAP ' + number(dc.get('rescale_only', {}).get('mean')),
             '95% CI ' + interval(dc.get('rescale_only')).split(' ', 1)[-1])], y=365)
        y = slides.para('三種改動都有個別改善與失敗；平均值不足以代表每一題。', y=230, size=18)
        slides.para('主提交在看 test 分數前固定為 Baseline，額外結果僅用於比較與分析。', y=y - 12, size=14, gray=True)
    link = args.artifact_url or '待補上公開成果資料夾連結'
    y = slides.para('程式、結果與重現說明', y=135, size=14)
    slides.para(link, y=y - 8, size=13)
    if args.artifact_url:
        slides.canvas.linkURL(args.artifact_url, (LEFT, 85, RIGHT, 138), relative=0)

    slides.start('方法｜檢索與生成分開處理', '使用 Stable Audio Open 1.0；自己的 CFG 實際接入 sampling loop。[1–3]')
    y = slides.para('caption → CLAP cosine 檢索；caption → 文字條件 → 去噪 → VAE 解碼 → 真正 MP3', y=495, size=18)
    y = slides.para('guided = uncond + scale × (cond − uncond)', y=y - 22, size=22)
    y = slides.para('scale = 0 是無文字條件，scale = 1 是文字條件預測；兩者仍有時長條件。generate.py 每步呼叫 cfg.py 的 guided_prediction，再交給 scheduler.step；模型預測 v。', y=y - 14, size=13.5)
    y = slides.table(['主提交設定', '實際紀錄'], [310, 702], y=y - 23, size=14, rows=[
        ['去噪／時長／CFG', f"{settings.get('steps', '待完成')} steps / {settings.get('seconds', '待完成')} 秒 / {settings.get('guidance_scale', '待完成')}"],
        ['裝置／精度／輸出', f"{settings.get('device', '待完成')} / {settings.get('dtype', '待完成')} / 44.1 kHz stereo MP3, {settings.get('mp3_bitrate', '待完成')}"],
        [scheduler_label, scheduler_details],
        ['獨立 pilot：官方流程＋同修正', f"最大 waveform 差 {number(parity.get('max_abs_difference'), 8)}；容許誤差 {number(parity.get('absolute_tolerance'), 8)}"]])
    seed_note = '各 ID 固定 CPU noise seed，不逐題找 seed。' if baseline_only else '各 ID 固定 CPU noise seed，跨條件使用相同初始 latent。'
    y = slides.para(f"{seed_note}正式生成每首耗時中位數 {number(summary['time_median'], 1)} 秒。", y=y - 13, size=12.5, gray=True)
    slides.para(compatibility_disclosure(data), y=y - 8, size=11.5, gray=True)

    slides.start('主結果｜完整 100 首的必要評估', '評估對準實際提交的 MP3；三種指標回答不同問題。[3–5,8]')
    y = slides.table(['指標', '數值', '參照與意義'], [300, 140, 572], y=495, rows=[
        ['FD-VGG（全部參照）', number(evaluation.get('FD-VGG-all-reference'), 3), 'VGGish 128 維視窗特徵的均值／共變異數距離；100 首生成 vs 5,367 首參照，低較接近。'],
        ['FD-VGG（檢索 proxy）', number(evaluation.get('FD-VGG-retrieved-proxy'), 3), '100 首生成 vs 100 筆 top-1 參照列；重複 ID 保留，另列結果。'],
        ['KL-PaSST（proxy 配對）', number(evaluation.get(KL)), 'PaSST 527 類 logits 做 softmax；KL(reference ∥ generated) 逐題平均，低較接近固定 top-1 proxy。'],
        ['CLAP', number(evaluation.get('CLAP')), '指定 CLAP 的文字／音訊 embedding 做 cosine 後逐題平均；高表示全局文字對齊較強。']], size=14)
    count = evaluation.get('counts', {}).get('unique_retrieved_references', '待完成')
    y = slides.para(f'100 筆配對中共有 {count} 個不同參照。檢索音訊是 proxy，並非新 caption 的 ground truth。', y=y - 23, size=15)
    y = slides.callout('兩個 FD 的參照集合不同，不能把 7.691 → 2.527 解讀成方法改善。', y=y - 16, size=15)
    slides.para('FD 彙整時間窗特徵，視窗不是獨立歌曲。KL 採檢索 proxy 配對；作業未明定此細節，尚無助教確認。', y=y - 12, size=12.5, gray=True)

    if not baseline_only:
        slides.start('設計｜兩因素、四條件，同一批 20 題', '主提交生成 100 題；消融只比較 ID 0–19，固定模型、步數、時長、noise、精度與編碼。')
        rows = []
        for condition in conditions:
            c = condition['settings']; name = condition['id']
            schedule = ('待凍結' if 'schedule' not in c else
                        f"線性 {c['guidance_scale']} → {c['end_scale']}" if c['schedule'] == 'linear' else '固定 7')
            rows.append([LABELS.get(name, name), schedule, str(c.get('rescale', '待完成')), '100 / 20' if name == 'baseline' else '20 / 20'])
        y = slides.table(['條件', 'Guidance 排程', '幅度修正 r', '生成／比較首數'], [270, 330, 210, 202], y=495, rows=rows, size=15)
        y = slides.para('s_i = 10 − 6i/(N−1),  i = 0,...,N−1;   g_i = u_i + s_i(c_i−u_i)', y=y - 15, size=16)
        y = slides.para('a_i = std(c_i)/std(g_i)  if std(g_i) > ε;  else a_i = 1', y=y - 7, size=16)
        y = slides.para("g'_i = (1−r)g_i + r a_i g_i,   r = 0.5", y=y - 7, size=16)
        y = slides.para('u/c 是無／有文字條件的 v 預測，仍保留時長條件。std 按每筆 channel/time 以 FP32、correction=0 計算，ε=float32 eps；r=0 原樣回傳。[6] N=100；兩排程步均值皆7，但非相同 sigma 積分或有效條件強度；diffusion scheduler 不變。', y=y - 12, size=13)
        slides.para('事前問題：改排程、幅度修正是否改變對齊／訊號？兩者是否交互影響？本 10→4 排程早期較強，與 [7] 圖像實驗偏好中段 guidance 不同，不能沿用其改善結論；這是待驗證的音訊假設。[6–7] ID 0–19 事先固定。', y=y - 10, size=12.5, gray=True)

        slides.start('配對結果｜CLAP 略升，區間仍包含 0', '同一批 20 題；95% percentile interval 以 caption 配對 bootstrap 計算，沒有把視窗當成獨立歌曲。')
        rows = []
        for name in LABELS:
            cell = ablation.get('summary', {}).get(name, {})
            dc = ablation.get('paired_deltas_vs_baseline', {}).get('CLAP', {}).get(name)
            dk = ablation.get('paired_deltas_vs_baseline', {}).get(KL, {}).get(name)
            rows.append([LABELS[name], number(cell.get('CLAP', {}).get('mean')),
                         '—' if name == 'baseline' else interval(dc), number(cell.get(KL, {}).get('mean')),
                         '—' if name == 'baseline' else interval(dk),
                         number(cell.get('FD-VGG-cohort-vs-retrieved-proxy'), 3)])
        y = slides.table(['條件', 'CLAP ↑', 'ΔCLAP [95% CI]', 'KL proxy ↓', 'ΔKL [95% CI]', 'FD proxy ↓'],
                         [155, 85, 255, 90, 255, 172], y=495, rows=rows, size=13.2)
        y = slides.para('Δ = 該條件 − Baseline。CLAP 正值、KL 負值分別是改善方向；FD 使用同一批 20 筆檢索 proxy。FD 是集合描述值，未做逐題 FD 或宣稱小樣本分布已穩定。', y=y - 18, size=14)
        wins = []
        for name in ('schedule_only', 'rescale_only', 'combined'):
            counts = paired_counts(data, name)
            wins.append(f"{LABELS[name]}：{counts['clap_up']}/20 上升、{counts['clap_down']}/20 下降" if counts else f'{LABELS[name]}：待完成')
        y = slides.para('逐題 ΔCLAP：' + '；'.join(wins) + '。其餘為持平。', y=y - 15, size=14)
        slides.para('僅前 20 個 ID 的探索性比較，不能把 20 題方法分數與完整 100 題 Baseline 相減。每題一個固定 seed，配對區間不估計跨 seed 變異，也未校正多重比較。', y=y - 16, size=13, gray=True)

        slides.start('消融｜尚未確認兩項改動的疊加效果', '四條件分辨排程、幅度修正，以及兩者合用是否超出各自變化的加總。')
        effects = ablation.get('factorial_effects', {})
        names = [('schedule_main_effect', '排程主效應'), ('rescale_main_effect', '幅度修正主效應'),
                 ('interaction_difference_in_differences', '交互作用：D − B − C + A')]
        y = slides.table(['比較', 'CLAP 差值 [95% CI]', 'KL proxy 差值 [95% CI]'], [300, 356, 356], y=495,
                         rows=[[label, interval(effects.get('CLAP', {}).get(key)), interval(effects.get(KL, {}).get(key))]
                               for key, label in names], size=13.5)
        y = slides.para('A = Baseline，B = 排程，C = 幅度修正，D = Combined。主效應平均另一因素的兩個水準；交互作用衡量兩者一起用時是否超出各自改變的加總。', y=y - 22, size=15)
        figure = paired_tradeoff_figure(data, args)
        if figure:
            # The plot and signal summary share the same vertical area; keep
            # the original ten-page scope and make heterogeneity visible.
            from reportlab.lib.utils import ImageReader
            top, height = y - 17, 195
            if top - height < CONTENT_BOTTOM:
                raise ValueError('Paired plot would overflow; revise layout')
            slides.canvas.drawImage(ImageReader(str(figure)), LEFT, top - height,
                                    width=490, height=height, preserveAspectRatio=True, anchor='c')
            right, width = 582, 484
            text_y = slides.para('實際 MP3：平均 RMS / peak / 超界比例', x=right, y=top, width=width, size=12.5)
            for name in LABELS:
                cell = ablation['summary'][name]
                rms, peak, clip = [cell[k]['mean'] for k in ('rms', 'peak', 'fraction_abs_ge_1')]
                text_y = slides.para(f'{LABELS[name]}：{rms:.4f} / {peak:.3f} / {100*clip:.3f}%',
                                     x=right, y=text_y - 8, width=width, size=12.5)
            slides.para('超界 = |sample| ≥ 1。這些是訊號診斷，不是失真或音樂性評分；圖中兩指標方向也不代表人類偏好。',
                        x=right, y=text_y - 9, width=width, size=11.8, gray=True)
        else:
            slides.para('逐題取捨圖及 RMS／peak／超界診斷，待完整配對結果後產生。', y=y - 30, size=15, gray=True)

    case_heading = 'Baseline 個案｜最低與中段分數' if baseline_only else '案例｜文字對齊與 proxy 接近度會取捨'
    case_subtitle = ('左欄最低 CLAP；右欄按 CLAP 排序第 50／100 的樣本。事後描述性選例，未回頭調參，也不是真人聽評。'
                     if baseline_only else 'Combined 的 ΔCLAP 最大／最小兩題；事後極端選例，非代表性樣本。中文僅摘要提示詞，生成仍用原始英文。')
    slides.start(case_heading, case_subtitle)
    slides.canvas.setStrokeColor(colors.HexColor('#dde3e9'))
    slides.canvas.setLineWidth(.6)
    slides.canvas.line(560, 495, 560, 95)
    low, changed = select_baseline_cases(data) if baseline_only else select_cases(data)
    x_positions, box_width = [LEFT, 582], 484
    for idx, case in enumerate((low, changed)):
        x = x_positions[idx]
        if case is None:
            slides.para('案例待正式結果完成後產生。', x=x, y=485, width=box_width, size=16)
            continue
        identifier = case['id']
        if baseline_only:
            title = f'ID {identifier}：全 100 首最低 CLAP' if idx == 0 else f'ID {identifier}：CLAP 排序第 50／100'
        else:
            title = f"ID {identifier}：Combined ΔCLAP {'最大' if idx == 0 else '最小'}"
        y = slides.para(title, x=x, y=492, width=box_width, size=19)
        caption = data['captions'][identifier]
        prompt_notes = {
            '4': '提示詞重點：輕鬆、簡約的 hip-hop，以電鋼琴、低音提琴、細微鈸聲營造柔和背景。',
            '8': '提示詞重點：復古 techno，清晰鼓聲、溫暖合成器鋪底與穩定低音線，帶有 1980 年代電子舞曲氛圍。'}
        if not baseline_only and identifier in prompt_notes:
            caption = prompt_notes[identifier]
        elif len(caption) > 420:
            caption = caption[:420].rsplit(' ', 1)[0] + ' …（原文節錄）'
        y = slides.para(caption, x=x, y=y - 14, width=box_width, size=15 if not baseline_only else 12.8)
        if baseline_only:
            record = data['records'].get(identifier, {})
            audit = record.get('audio_audit', {})
            evidence = f"CLAP {number(case['clap'])}；KL proxy {number(case.get('kl_passt_softmax'))}。\nMP3 RMS {number(audit.get('rms'))}；peak {number(audit.get('peak'), 3)}。\n檢索 reference：{case.get('reference_id', '待完成')}"
            interpretation = ('這一例顯示全局對齊相對較弱。分數不能指出究竟漏了哪個樂器或節奏；要確認具體音樂內容，仍需人工核聽。'
                              if idx == 0 else '這是同一 Baseline 中接近中段的分數，兩例的 caption 不同，因此分數差不是方法改善的證據，也不能當作人類品質排名。')
        else:
            before = data['ablation_rows'].get(('baseline', identifier), {})
            evidence = (f"CLAP {number(before.get('CLAP'))} → {number(case.get('CLAP'))}（Δ {number(case.get('CLAP_delta_vs_baseline'))}）\n"
                        f"KL proxy {number(before.get(KL))} → {number(case.get(KL))}（Δ {number(case.get(KL + '_delta_vs_baseline'))}）\n"
                        f"RMS {number(before.get('rms'))} → {number(case.get('rms'))}；peak {number(before.get('peak'), 3)} → {number(case.get('peak'), 3)}\n"
                        f"檢索 reference：{case.get('reference_id', '待完成')}")
            dc, dk = case['CLAP_delta_vs_baseline'], case.get(KL + '_delta_vs_baseline')
            if finite(dk) and dc > 0 and dk < 0:
                interpretation = '此題文字對齊與 proxy 接近度同時改善；極端案例不代表其他 captions 或 seeds。'
            elif finite(dk) and dc < 0 and dk > 0:
                interpretation = '此題文字對齊與 proxy 接近度同時下降，是額外方法的具體失敗例；沒有因此換 seed 或修改方法。'
            elif finite(dk) and dc * dk > 0:
                interpretation = '此題兩指標呈現取捨：更靠近文字，未必更靠近檢索音訊；proxy 不等於正確答案。'
            else:
                interpretation = '至少一指標持平，無法稱為兩指標一致改善；仍須看全部配對結果與區間。'
            if identifier == '4':
                interpretation = '文字對齊上升，但 KL 變高、離 proxy 更遠。CLAP 的上升仍不能逐項證明電鋼琴、低音提琴等概念都正確。'
            elif identifier == '8':
                before_clip, after_clip = before['fraction_abs_ge_1'], case['fraction_abs_ge_1']
                interpretation = (f'超界比例由 {100*before_clip:.3f}% 降至 {100*after_clip:.3f}%，CLAP 卻下降；'
                                  '訊號幅度與文字對齊的方向不同。這裡的判讀來自指標，不是真人聽評。')
        y = slides.para(evidence, x=x, y=y - 18, width=box_width, size=16 if not baseline_only else 13.5)
        slides.para(interpretation, x=x, y=y - 20, width=box_width, size=14, gray=True)

    slides.start('診斷｜平均通過，不代表每題都通過', '指定 music_audioset checkpoint，HTSAT-base／fusion=False；沒有使用 evaluator 的另一個預設權重。')
    pairing_control = evaluation.get('clap_pairing_control', {})
    silence_control = evaluation.get('clap_silence_control', {})
    t5_truncated = summary['t5_truncated'] if data['records'] else '待完成'
    diagnostics = control_diagnostics(data)
    mismatch_count = len(diagnostics['mismatch_not_better_ids']) if diagnostics else '待完成'
    silence_count = len(diagnostics['silence_not_better_ids']) if diagnostics else '待完成'
    y = slides.table(['實際稽核', '結果', '應如何解讀'], [290, 180, 542], y=495, size=13.5, rows=[
        ['100 次 top-1 檢索', f"{meta.get('unique_retrieved_references', '待完成')} 個不同 ID", '同一首可成為多題的最佳 proxy；唯一數不是 retrieval accuracy。'],
        ['CLAP 77 tokens 截斷', f"{meta.get('captions_truncated', '待完成')} / 100", '原始 caption 保留，但超出 token 上限的尾端概念不會進入 CLAP。'],
        ['生成 tokenizer 截斷', f'{t5_truncated} / 100', '生成與 CLAP 的 tokenizer 不同，因此它們看到的資訊可能不同。'],
        ['CLAP 負控制平均', f"錯配 {number(pairing_control.get('cyclic_mismatch_mean'))}\n靜音 {number(silence_control.get('silence_mean'))}",
         f'生成分數不高於錯配者 {mismatch_count}/100；不高於靜音者 {silence_count}/100。平均通過，不代表每題都通過。']])
    y = slides.para('CLAP 前處理：mono、去 DC、48 kHz、固定 10 秒；VGGish 用 16 kHz、PaSST 用 32 kHz，各依原評估流程。', y=y - 17, size=13.5)
    low = min(data['per_score'].values(), key=lambda r: r['clap']) if data['per_score'] else None
    if low and finite(low.get('clap_silence')):
        truncated = details.get(low['id'], {}).get('tokenizer', {}).get('was_truncated')
        token_fact = '被截斷' if truncated is True else '未截斷' if truncated is False else '狀態待查'
        y = slides.para(f"最低分 ID {low['id']}：生成 {number(low['clap'])} < 靜音 {number(low['clap_silence'])}，CLAP 文字{token_fact}。因此不能把所有低分都歸因於截斷，也不能把這個分數差當作靜音較好聽。", y=y - 13, size=13.5)
        if low['id'] == '35':
            y = slides.para('ID 35 同時要求鼓機、切分節奏，以及旅行等抽象意象。全局 CLAP 無法逐項判定哪個概念未對齊；這是需要細粒度概念評估的理由，不能從低分推定漏了某個樂器。', y=y - 12, size=12.5)
    slides.para('CLAP 同時用於檢索與評估，兩者不是獨立證據。錯配 caption 仍可能共享音樂概念；靜音只是一種負音訊，不能校準人類品質。', y=y - 12, size=12.5, gray=True)

    slides.start('討論｜結果支持什麼，還缺什麼證據')
    discussion = [
        combined_conclusion(ablation),
        '四條件共用每題的初始 noise，減少抽樣差異；平均 guidance 同為 7，讓比較更聚焦在步驟安排。但 step 平均相同，不等於對 sigma 的積分或條件向量幅度相同。',
        '只有固定前 20 題、每題一個 seed，且做多個探索性比較。配對區間描述這一組樣本，不能保證其他 captions、seeds 或裝置也有相同效果。',
        '沒有做人類偏好測試；MQAScore 為選做，本次沒有把自由文字音訊評論冒充它。下一步可增加獨立開發題、多種 seed 與盲聽，再確認指標中的差異。']
    if baseline_only:
        discussion = [
            '本版呈現完成的四項基本工作：CLAP 檢索、自己的 CFG、100 首生成與必要評估。排程／幅度修正的額外實驗尚未完成驗證，不列入本版結果，也不聲稱改善。',
            'FD 比較整批特徵分布；KL 比較與檢索 proxy 的類別分布；CLAP 比較全局文字對齊。三者不能互換解讀，檢索 proxy 也不是 ground truth。',
            '循環錯配與靜音控制只檢查 CLAP 對配對及音訊內容的粗略反應。即使生成分數較高，也不能證明每個樂器、節奏或否定詞都正確。',
            '每題只有一個固定 seed，沒有真人偏好測試；MQAScore 是選做，本版未實作。後續可完成預宣告消融，增加獨立開發題、多 seed 與盲聽；不得由本版 test 分數回頭調參。']
    blocks = None if baseline_only else discussion_blocks(data)
    if blocks:
        y = 500
        for title, body in blocks:
            y = slides.para(title, y=y, size=18)
            y = slides.para(body, y=y - 7, size=15, leading=21) - 19
    else:
        if not baseline_only:
            discussion = analysis_discussion(data)
        slides.bullets(discussion, y=500, size=14.5 if not baseline_only else 16, gap=15)

    slides.start('重現與來源｜程式、紀錄、引用', '完整原碼、固定版本與逐首紀錄放在成果資料夾；正文保留理解結果需要的資訊。')
    flow = ('主流程：官方授權模型下載 → 獨立提示詞試跑 → 凍結設定 → 檢索／生成 → 對實際 MP3 評估 → ZIP 驗證。'
            if baseline_only else '主流程：官方授權模型下載 → 獨立提示詞試跑 → 凍結設定 → 檢索／生成 → 對實際 MP3 評估 → 配對分析 → ZIP 驗證。')
    y = slides.para(flow, y=495, size=16)
    y = slides.para('未自行訓練 checkpoint；提供官方固定 revision 與下載／推論指令，排除資料集與登入 cache。', y=y - 14, size=14)
    sources = [
        ('[1] Stable Audio Open 1.0 @ f21265c1', 'https://huggingface.co/stabilityai/stable-audio-open-1.0/tree/f21265c1e2710b3bd2386596943f0007f55f802e', None, None),
        ('[2] Diffusers 0.41.0 StableAudioPipeline', 'https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/pipelines/stable_audio/pipeline_stable_audio.py', None, None),
        ('[3] LAION-CLAP 程式', 'https://github.com/LAION-AI/CLAP',
         '權重：music_audioset_epoch_15_esc_90.14.pt', 'https://huggingface.co/lukewys/laion_clap/blob/b3708341862f581175dba5c356a4ebf74a9b6651/music_audioset_epoch_15_esc_90.14.pt'),
        ('[4] av-benchmark @ f351b9a6', 'https://github.com/hkchengrex/av-benchmark/tree/f351b9a6fc6abde746d5f8e1d4c47c883319cb41', None, None),
        ('[5] hear21passt 程式 @ 1ab77898', 'https://github.com/hkchengrex/passt_hear21/tree/1ab778984f67cca09dd8bbad11f728a0122cce24',
         '權重：passt-s-f128-p16-s10-ap.476-swa.pt', 'https://github.com/kkoutini/PaSST/releases/tag/v0.0.1-audioset'),
        ('[6] Lin et al., Common Diffusion Noise Schedules and Sample Steps are Flawed', 'https://arxiv.org/abs/2305.08891', None, None),
        ('[7] Kynkäänniemi et al., Applying Guidance in a Limited Interval…', 'https://arxiv.org/abs/2404.07724', None, None),
        ('[8] torchvggish v0.1 程式／權重：vggish-10086976.pth', 'https://github.com/harritaylor/torchvggish/releases/tag/v0.1', None, None)]
    if baseline_only:
        sources = [row for row in sources if not row[0].startswith(('[6]', '[7]'))]
    y -= 22
    for title, url, weight_title, weight_url in sources:
        start = y
        left_width = 330 if weight_title else RIGHT - LEFT
        left_bottom = slides.para(title, y=y, width=left_width, size=12.8)
        slides.canvas.linkURL(url, (LEFT, left_bottom, LEFT + left_width, start), relative=0)
        if weight_title:
            right_x = LEFT + 350
            right_bottom = slides.para(weight_title, x=right_x, y=y, width=RIGHT - right_x, size=12.8)
            slides.canvas.linkURL(weight_url, (right_x, right_bottom, RIGHT, start), relative=0)
            y = min(left_bottom, right_bottom) - 7
        else:
            y = left_bottom - 7
    y = slides.para('VGGish／PaSST 經固定 HF 鏡像傳輸；完整 SHA 對應鏡像，大小對照原 release。舊 release 沒有 full digest，因此未宣稱獨立官方完整 SHA 驗證。鏡像來源、版本與核對紀錄見公開資料夾。', y=y - 7, size=12, gray=True)
    y = slides.para('作業依據：Homework_3.pdf、sample code 與 COOL 公告。AI 協助程式、整理與編排；數值取自執行紀錄，未用提供的 validation／test 調參或逐題找 seed。', y=y - 9, size=12.5, gray=True)
    if args.draft:
        slides.para(f"尚有 {len(data['issues'])} 類驗證事項未完成；此檔不是正式提交報告。", y=y - 10, size=12, gray=True)
    slides.save()


def build(args):
    baseline_only = getattr(args, 'baseline_only', False)
    report_scope = 'baseline_only' if baseline_only else 'full_with_ablation'
    expected_pages = 7 if baseline_only else 10
    if args.draft:
        if args.output is None:
            args.output = ROOT / ('analysis/r14942154_report_BASELINE_DRAFT.pdf' if baseline_only else
                                  'analysis/r14942154_report_DRAFT.pdf')
        if args.output.name == STUDENT + '_report.pdf' or 'draft' not in args.output.stem.lower():
            raise ValueError('Draft output must include DRAFT and must not use the official report filename')
    elif args.output is None:
        args.output = ROOT / (STUDENT + '_report.pdf')
    if args.output.suffix.lower() != '.pdf':
        raise ValueError('Report output must be a PDF')
    if not args.font.is_file():
        raise ValueError('Provide a local TrueType font covering Traditional Chinese via --font')
    pdfmetrics.registerFont(TTFont(FONT, str(args.font)))
    data = validate_bundle(args)
    args.figure_dir = ROOT / 'analysis/report_review/figures'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.hw3-report-', dir=args.output.parent) as temporary:
        working = Path(temporary) / 'report.pdf'
        render(Slides(working, args.draft, total_pages=expected_pages), data, args)
        from pypdf import PdfReader
        reader = PdfReader(working)
        if len(reader.pages) != expected_pages:
            raise ValueError(f'The {report_scope} report must contain {expected_pages} pages')
        for page in reader.pages:
            ratio = float(page.mediabox.width) / float(page.mediabox.height)
            if abs(ratio - 16 / 9) > 1e-9:
                raise ValueError('Every report page must be 16:9')
            extracted = page.extract_text()
            if STUDENT not in extracted or (args.draft and '草稿' not in extracted):
                raise ValueError('PDF text is missing the student ID or draft marker')
        working.replace(args.output)
    sources = {k: {'path': str(v), 'sha256': sha(v)} for k, v in data['paths'].items() if v.is_file()}
    case_policy = ('Post-hoc descriptive baseline cases: minimum full-100 CLAP, and rank 50 of 100 by CLAP. Different captions; not a method comparison.'
                   if baseline_only else 'Post-hoc descriptive cases: maximum and minimum combined-minus-baseline CLAP within all 20 fixed IDs. Extremes are not representative; no parameter, seed or submission selection.')
    result = {'report': str(args.output), 'sha256': sha(args.output), 'pages': expected_pages, 'aspect': '16:9',
              'report_scope': report_scope, 'ablation_included': not baseline_only,
              'draft': args.draft, 'formal_result_validation_passed': not data['issues'],
              'artifact_url': args.artifact_url, 'artifact_public_access_verified_by_builder': False,
              'remaining_issues': data['issues'], 'inputs': sources,
              'builder_path': str(Path(__file__).resolve().relative_to(ROOT)),
              'builder_sha256': sha(__file__), 'font_sha256': sha(args.font),
              'case_selection': case_policy,
              'descriptive_analysis': {
                  'policy': 'Post-hoc report-only descriptive summaries, not new metrics or test tuning.',
                  'baseline_controls': control_diagnostics(data),
                  'paired_counts': {n: paired_counts(data, n) for n in LABELS if n != 'baseline'}}}
    args.output.with_suffix('.checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'inputs', 'remaining_issues'}}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation-dir', type=Path, default=ROOT / 'results/baseline')
    p.add_argument('--evaluation', type=Path, default=ROOT / 'results/baseline_evaluation.json')
    p.add_argument('--ablation', type=Path, default=ROOT / 'results/ablation/ablation_results.json')
    p.add_argument('--retrieval-dir', type=Path, default=ROOT / 'results/retrieval')
    p.add_argument('--protocol', type=Path, default=ROOT / 'analysis/experiment_protocol_frozen.json')
    p.add_argument('--captions', type=Path, default=ROOT / 'sample_code/test.csv')
    p.add_argument('--artifact-url')
    p.add_argument('--font', type=Path, default=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    p.add_argument('--draft', action='store_true')
    p.add_argument('--baseline-only', action='store_true',
                   help='Formal seven-page completed-baseline scope; all baseline checks remain required, ablation results are excluded')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    try:
        build(a)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        p.exit(1, 'Report build stopped: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()
