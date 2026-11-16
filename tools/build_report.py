"""Build a plain Chinese 16:9 HW3 report from recorded, real experiment outputs.

No score is imputed. Final mode rejects missing/inconsistent results or a missing
artifact URL. --draft visibly marks every page and cannot use the official name.
Run with .venv_report/bin/python; generation/evaluation environments stay separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
                               textColor=colors.HexColor('#555555') if gray else colors.black,
                               wordWrap='CJK', allowWidows=0, allowOrphans=0)
        markup = escape(str(text)).replace('\n', '<br/>')
        p = Paragraph(markup, style)
        _, height = p.wrap(width, HEIGHT)
        if y - height < (limit or CONTENT_BOTTOM):
            raise ValueError(f'Report page {self.page} overflows; shorten the content instead of hiding it')
        p.drawOn(self.canvas, x, y - height)
        return y - height

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
        style = ParagraphStyle('cell', fontName=FONT, fontSize=size, leading=size * 1.35, wordWrap='CJK')
        for index, row in enumerate([headers] + rows):
            paragraphs, heights = [], []
            for value, width in zip(row, widths):
                p = Paragraph(escape(str(value)).replace('\n', '<br/>'), style)
                heights.append(p.wrap(width - 18, HEIGHT)[1]); paragraphs.append(p)
            height = max(heights) + 22
            if y - height < CONTENT_BOTTOM:
                raise ValueError(f'Table on page {self.page} overflows')
            if index == 0:
                c.setFillColor(colors.HexColor('#eeeeee')); c.rect(LEFT, y - height, sum(widths), height, fill=1, stroke=0)
            x = LEFT
            for p, width, h in zip(paragraphs, widths, heights):
                p.drawOn(c, x + 9, y - 11 - h); x += width
            c.setStrokeColor(colors.HexColor('#b7b7b7')); c.setLineWidth(.45)
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
        c.setStrokeColor(colors.HexColor('#777777')); c.setLineWidth(.6)
        c.line(LEFT, HEIGHT - 90, RIGHT, HEIGHT - 90)
        if subtitle:
            self.para(subtitle, y=HEIGHT - 104, size=12, gray=True)
        c.setFillColor(colors.HexColor('#666666')); c.setFont(FONT, 10)
        c.drawString(LEFT, 27, 'HW3  |  ' + STUDENT)
        c.drawRightString(RIGHT, 27, str(self.page) + ' / ' + str(self.total_pages))
        if self.draft:
            c.setFont(FONT, 14); c.drawCentredString(WIDTH / 2, 25, '草稿：實驗／公開連結尚未全部驗證，不可提交')

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
    scores = [r for r in data['per_score'].values() if finite(r.get('clap'))]
    low = min(scores, key=lambda r: (r['clap'], int(r['id']))) if scores else None
    candidates = [r for r in data['ablation_rows'].values()
                  if r.get('condition') != 'baseline' and finite(r.get('CLAP_delta_vs_baseline'))
                  and (low is None or r['id'] != low['id'])]
    changed = max(candidates, key=lambda r: (abs(r['CLAP_delta_vs_baseline']), -int(r['id']), r['condition'])) if candidates else None
    return low, changed


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

    slides.start('HW3：文字到音樂 Baseline 成果' if baseline_only else 'HW3：文字到音樂與 CFG 對照實驗')
    slides.para('r14942154', y=480, size=23)
    slides.para('完成 100 筆檢索、CFG、生成與必要評估。' if baseline_only else
                '先完成完整生成，再用同一批提示詞比較 guidance 的安排方式。', y=425, size=22)
    cover_points = [
        '主結果：100 個原始 captions 的 CLAP 檢索、音樂生成與三種必要評估。',
        '額外實驗：預先固定前 20 個 ID，做排程 × 幅度修正的四條件配對比較。',
        '主提交方法在看 test 分數前固定為 Baseline；對照結果不拿來挑提交音訊。']
    if baseline_only:
        cover_points = [cover_points[0],
                        '本版只呈現已完成的 Baseline；額外實驗尚未完成驗證，不列為成果。',
                        '方法與種子在看 test 分數前固定；負控制及個案分析不作人類品質聲稱。']
    slides.bullets(cover_points, y=355, size=17, gap=18)
    link = args.artifact_url or '待補上公開成果資料夾連結'
    y = slides.para('程式、模型固定來源與執行說明：', y=175, size=14)
    slides.para(link, y=y - 8, size=13)
    if args.artifact_url:
        slides.canvas.linkURL(args.artifact_url, (LEFT, 95, RIGHT, 156), relative=0)

    slides.start('1. 方法：檢索、CFG、生成', '使用 Stable Audio Open 1.0；自己的 CFG 確實接進 sampling loop。[1–3]')
    y = slides.para('caption → CLAP cosine 檢索；caption → 文字條件 → 去噪 → VAE 解碼 → 真正 MP3', y=495, size=18)
    y = slides.para('guided = uncond + scale × (cond − uncond)', y=y - 22, size=22)
    y = slides.para('scale = 0 是無條件，scale = 1 是條件預測。模型實際預測 v；範例的 noise_pred 只是變數名。', y=y - 14, size=15)
    y = slides.table(['主提交設定', '實際紀錄'], [310, 702], y=y - 23, size=14, rows=[
        ['去噪／時長／CFG', f"{settings.get('steps', '待完成')} steps / {settings.get('seconds', '待完成')} 秒 / {settings.get('guidance_scale', '待完成')}"],
        ['裝置／精度／輸出', f"{settings.get('device', '待完成')} / {settings.get('dtype', '待完成')} / 44.1 kHz stereo MP3, {settings.get('mp3_bitrate', '待完成')}"],
        [scheduler_label, scheduler_details],
        ['獨立 pilot：官方流程＋同修正', f"最大 waveform 差 {number(parity.get('max_abs_difference'), 8)}；容許誤差 {number(parity.get('absolute_tolerance'), 8)}"]])
    seed_note = '各 ID 固定 CPU noise seed，不逐題找 seed。' if baseline_only else '各 ID 固定 CPU noise seed，跨條件使用相同初始 latent。'
    y = slides.para(f"{seed_note}正式生成每首耗時中位數 {number(summary['time_median'], 1)} 秒。", y=y - 13, size=12.5, gray=True)
    slides.para(compatibility_disclosure(data), y=y - 8, size=11.5, gray=True)

    slides.start('2. 完整 100 首結果', '評估對準實際提交的 MP3；三種指標回答不同問題。[3–5]')
    y = slides.table(['指標', '數值', '參照與意義'], [300, 140, 572], y=495, rows=[
        ['FD-VGG（全部參照）', number(evaluation.get('FD-VGG-all-reference'), 3), '100 首生成 vs 5,367 首 MusicCaps；整體特徵分布，低較接近。'],
        ['FD-VGG（檢索 proxy）', number(evaluation.get('FD-VGG-retrieved-proxy'), 3), '100 首生成 vs 100 筆 top-1 參照列；重複 ID 保留，另列結果。'],
        ['KL-PaSST（proxy 配對）', number(evaluation.get(KL)), 'KL(reference ∥ generated)；每題固定 top-1 配對，低較接近 proxy。'],
        ['CLAP', number(evaluation.get('CLAP')), '原始 caption 與生成音訊的平均 cosine；高表示全局對齊較強。']], size=14)
    count = evaluation.get('counts', {}).get('unique_retrieved_references', '待完成')
    y = slides.para(f'100 筆配對中共有 {count} 個不同參照。檢索音訊是 proxy，並非新 caption 的 ground truth。', y=y - 23, size=15)
    slides.para('作業沒有明定 FD 的參照集合與 KL 配對。這裡把兩種 FD 分開，KL 明示採用檢索 proxy，沒有假稱這是助教確認的設定。', y=y - 16, size=14, gray=True)

    if not baseline_only:
        slides.start('3. 額外實驗：哪些改動真的造成差異？', '四條件採同一批 caption、模型、步數、時長、noise、精度與 MP3 編碼。')
        rows = []
        for condition in conditions:
            c = condition['settings']; name = condition['id']
            schedule = ('待凍結' if 'schedule' not in c else
                        f"線性 {c['guidance_scale']} → {c['end_scale']}" if c['schedule'] == 'linear' else '固定 7')
            rows.append([LABELS.get(name, name), schedule, str(c.get('rescale', '待完成')), '100' if name == 'baseline' else '20'])
        y = slides.table(['條件', 'Guidance 排程', '幅度修正 r', '規劃首數'], [270, 330, 210, 202], y=495, rows=rows, size=15)
        y = slides.para('s_i = 10 − 6i/(N−1),  i = 0,...,N−1;   g_i = u_i + s_i(c_i−u_i)', y=y - 15, size=16)
        y = slides.para('a_i = std(c_i)/std(g_i)  if std(g_i) > ε;  else a_i = 1', y=y - 7, size=16)
        y = slides.para("g'_i = (1−r)g_i + r a_i g_i,   r = 0.5", y=y - 7, size=16)
        y = slides.para('u/c 是無／有文字條件的 v 預測，仍保留 duration 條件；g 是 CFG 結果。std 沿每筆 channel/time 維度以FP32、correction=0計算，ε=float32 eps；r=0原封回傳CFG。[6] 這是 CFG 強度排程，未替換 diffusion scheduler；N=100，兩種排程平均皆為7。', y=y - 12, size=13)
        slides.para('前強後弱與 prediction 幅度修正，是待驗證的音訊假設；不是直接重現 limited-interval 論文，也不保證音樂變好。[7] ID 0–19 在看 test 分數前選好。', y=y - 10, size=12.5, gray=True)

        slides.start('4. 配對結果：只比較同一批 20 個 ID', '表中區間是配對 caption bootstrap 的 95% percentile interval；1 首音訊不是多個獨立樣本。')
        rows = []
        for name in LABELS:
            cell = ablation.get('summary', {}).get(name, {})
            dc = ablation.get('paired_deltas_vs_baseline', {}).get('CLAP', {}).get(name)
            dk = ablation.get('paired_deltas_vs_baseline', {}).get(KL, {}).get(name)
            rows.append([LABELS[name], number(cell.get('CLAP', {}).get('mean')),
                         '—' if name == 'baseline' else interval(dc), number(cell.get(KL, {}).get('mean')),
                         '—' if name == 'baseline' else interval(dk)])
        y = slides.table(['條件', 'CLAP', 'ΔCLAP [95% CI]', 'KL proxy', 'ΔKL [95% CI]'],
                         [170, 100, 302, 110, 330], y=495, rows=rows, size=12.8)
        y = slides.para('Δ = 該條件 − Baseline。CLAP 的正值表示對齊較強；KL 的負值表示更接近檢索 proxy，兩者不能互換解讀。', y=y - 25, size=16)
        slides.para('這是前 20 個 ID 的探索性結果，不能當成完整 100 首的分數。每題只有一個固定 seed，也未做多重比較校正；區間不代表人類聽感或不同種子的穩定性。', y=y - 18, size=14, gray=True)

        slides.start('5. 拆開主效應與交互作用', '單看 combined，分不出改善來自哪個改動；四條件讓這個問題可以檢查。')
        effects = ablation.get('factorial_effects', {})
        names = [('schedule_main_effect', '排程主效應'), ('rescale_main_effect', '幅度修正主效應'),
                 ('interaction_difference_in_differences', '交互作用：D − B − C + A')]
        y = slides.table(['比較', 'CLAP 差值 [95% CI]', 'KL proxy 差值 [95% CI]'], [300, 356, 356], y=495,
                         rows=[[label, interval(effects.get('CLAP', {}).get(key)), interval(effects.get(KL, {}).get(key))]
                               for key, label in names], size=13.5)
        y = slides.para('A = Baseline，B = 排程，C = 幅度修正，D = Combined。主效應平均另一因素的兩個水準；交互作用衡量兩者一起用時是否超出各自改變的加總。', y=y - 22, size=15)
        signal_rows = []
        for metric, label in [('rms', '平均 RMS'), ('peak', '平均 peak'),
                              ('fraction_abs_ge_1', '平均 |sample| ≥ 1 比例')]:
            row = [label]
            for name in LABELS:
                value = ablation.get('summary', {}).get(name, {}).get(metric, {}).get('mean')
                row.append(number(100 * value if finite(value) else None, 3) + '%'
                           if metric == 'fraction_abs_ge_1' else number(value, 4 if metric == 'rms' else 3))
            signal_rows.append(row)
        y = slides.table(['訊號診斷'] + list(LABELS.values()), [280, 183, 183, 183, 183],
                         y=y - 19, rows=signal_rows, size=11.5)
        slides.para('幅度／超界比例只能提示訊號變化，不能直接當作失真、音樂性或音量偏好的證明。', y=y - 13, size=12, gray=True)

    case_heading = '3. Baseline 個案：最低與中段分數' if baseline_only else '6. 具體案例：用可核對的證據談取捨'
    case_subtitle = ('左欄最低 CLAP；右欄按 CLAP 排序第 50／100 的樣本。事後描述性選例，未回頭調參，也不是真人聽評。'
                     if baseline_only else '右欄：Baseline → 改動後。兩例為事後描述性選例，未回頭調參；指標分析不當作真人聽評。')
    slides.start(case_heading, case_subtitle)
    low, changed = select_baseline_cases(data) if baseline_only else select_cases(data)
    x_positions, box_width = [LEFT, 582], 484
    for idx, case in enumerate((low, changed)):
        x = x_positions[idx]
        if case is None:
            slides.para('案例待正式結果完成後產生。', x=x, y=485, width=box_width, size=16)
            continue
        identifier = case['id']
        title = (f'ID {identifier}：全 100 首最低 CLAP' if idx == 0 else
                 f'ID {identifier}：CLAP 排序第 50／100' if baseline_only else
                 f"ID {identifier}：配對變化最大（{LABELS.get(case['condition'], case['condition'])}）")
        y = slides.para(title, x=x, y=492, width=box_width, size=17)
        y = slides.para(data['captions'][identifier], x=x, y=y - 14, width=box_width, size=12.8)
        if idx == 0 or baseline_only:
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
            tradeoff = finite(dk) and dc * dk > 0
            interpretation = ('CLAP 與 proxy KL 在這一題呈現取捨：靠近文字的方向，和靠近檢索音訊的方向沒有一致。'
                              if tradeoff else '這一題對 CFG 改動較敏感，但單一案例不能代表整個子集；仍要一起看配對平均與區間。')
        y = slides.para(evidence, x=x, y=y - 18, width=box_width, size=13.5)
        slides.para(interpretation, x=x, y=y - 20, width=box_width, size=14, gray=True)

    slides.start(('4.' if baseline_only else '7.') + ' 檢索與文字截斷：分數也有盲點', '指定 music_audioset checkpoint，HTSAT-base／fusion=False；沒有使用 evaluator 的另一個預設權重。')
    pairing_control = evaluation.get('clap_pairing_control', {})
    silence_control = evaluation.get('clap_silence_control', {})
    t5_truncated = summary['t5_truncated'] if data['records'] else '待完成'
    y = slides.table(['實際稽核', '結果', '應如何解讀'], [290, 180, 542], y=495, size=14, rows=[
        ['100 次 top-1 檢索', f"{meta.get('unique_retrieved_references', '待完成')} 個不同 ID", '同一首可成為多題的最佳 proxy；唯一數不是 retrieval accuracy。'],
        ['CLAP 77 tokens 截斷', f"{meta.get('captions_truncated', '待完成')} / 100", '原始 caption 保留，但超出 token 上限的尾端概念不會進入 CLAP。'],
        ['生成 tokenizer 截斷', f'{t5_truncated} / 100', '生成與 CLAP 的 tokenizer 不同，因此它們看到的資訊可能不同。'],
        ['CLAP 負控制', f"錯配 {number(pairing_control.get('cyclic_mismatch_mean'))}\n靜音 {number(silence_control.get('silence_mean'))}",
         '同一批 captions 的循環錯配與 10 秒全零音訊；檢查評估器的粗略反應，不是人類品質校準。']])
    y = slides.para('每首音訊固定轉 mono、去 DC、重採樣至 48 kHz，再裁切／補齊為 480,000 點；避免隱含 random crop，並記錄原始長度和截斷。', y=y - 24, size=15)
    slides.para(f"生成相對錯配／靜音的平均 CLAP 差為 {number(pairing_control.get('matched_minus_mismatch_mean'))}／{number(silence_control.get('generated_minus_silence_mean'))}。CLAP 同時用於檢索與評估，兩者不是獨立證據；即使通過負控制，也不能證明每個樂器細節都正確。", y=y - 17, size=14, gray=True)

    slides.start(('5.' if baseline_only else '8.') + ' 討論：哪些結論能說，哪些還不行')
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
    slides.bullets(discussion, y=500, size=16, gap=18)

    slides.start(('6.' if baseline_only else '9.') + ' 重現方式與來源', '完整原碼、固定版本和逐首紀錄放在公開成果資料夾；主頁只保留理解結果需要的資訊。')
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
        left_bottom = slides.para(title, y=y, width=left_width, size=11.6)
        slides.canvas.linkURL(url, (LEFT, left_bottom, LEFT + left_width, start), relative=0)
        if weight_title:
            right_x = LEFT + 350
            right_bottom = slides.para(weight_title, x=right_x, y=y, width=RIGHT - right_x, size=11.6)
            slides.canvas.linkURL(weight_url, (right_x, right_bottom, RIGHT, start), relative=0)
            y = min(left_bottom, right_bottom) - 7
        else:
            y = left_bottom - 7
    y = slides.para('VGGish／PaSST 經固定 HF 鏡像傳輸；完整 SHA 對應鏡像，大小對照原 release。舊 release 沒有 full digest，因此未宣稱獨立官方完整 SHA 驗證。鏡像來源、版本與核對紀錄見公開資料夾。', y=y - 7, size=11.2, gray=True)
    y = slides.para('作業依據：Homework_3.pdf、sample code 與 COOL 公告。AI 協助程式、整理與編排；數值取自執行紀錄，未用提供的 validation／test 調參或逐題找 seed。', y=y - 9, size=11.5, gray=True)
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
                   if baseline_only else 'Post-hoc descriptive cases: minimum full-100 CLAP, then largest absolute paired CLAP change at another ID. No parameter or submission selection.')
    result = {'report': str(args.output), 'sha256': sha(args.output), 'pages': expected_pages, 'aspect': '16:9',
              'report_scope': report_scope, 'ablation_included': not baseline_only,
              'draft': args.draft, 'formal_result_validation_passed': not data['issues'],
              'artifact_url': args.artifact_url, 'artifact_public_access_verified_by_builder': False,
              'remaining_issues': data['issues'], 'inputs': sources,
              'builder_sha256': sha(__file__), 'font_sha256': sha(args.font),
              'case_selection': case_policy}
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
