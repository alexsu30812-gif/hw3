"""Display live HW3 progress without changing the experiment or artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def count(name):
    data = read(ROOT / 'results' / name / 'manifest.json')
    return len(data.get('samples', [])), data


def alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


def render():
    now = datetime.now().astimezone()
    supervisor = read(ROOT / 'analysis/resume_supervisor_state.json')
    pipeline = read(ROOT / 'analysis/pipeline_state_v6.json')
    full_finalize = read(ROOT / 'analysis/finalization_state.json')
    destination = read(ROOT / 'analysis/artifact_destination.json')
    retrieval = read(ROOT / 'results/retrieval/retrieval_manifest.json')
    baseline, manifest = count('baseline')
    stages = pipeline.get('stages', {})
    labels = {
        'vggish_mps_probe': '評估器檢查', 'freeze': '固定實驗設定',
        'retrieval': '100 題音樂檢索', 'generate_baseline': '生成必交音檔',
        'reference_vggish': '計算參照音樂 VGGish 特徵',
        'reference_passt': '計算參照音樂 PaSST 特徵',
        'baseline_vggish': '計算生成音檔 VGGish 特徵',
        'baseline_passt': '計算生成音檔 PaSST 特徵',
        'required_metrics': '計算必要評估指標', 'cool_zip': '檢查並打包 COOL ZIP',
        'baseline_cool_artifacts': '建立必交報告、備份及公開成果',
        'generate_schedule_only': '額外實驗：排程',
        'generate_rescale_only': '額外實驗：幅度修正',
        'generate_combined': '額外實驗：組合',
        'paired_ablation': '分析額外實驗', 'report_draft': '檢查加強報告',
    }
    stage = pipeline.get('current_stage', '尚未開始')
    running = alive(supervisor.get('pid')) and supervisor.get('status') == 'running'
    beat = supervisor.get('heartbeat_at_utc')
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(beat)).total_seconds()
    except (ValueError, TypeError):
        age = None
    if pipeline.get('status') == 'stopped' or supervisor.get('status') == 'stopped':
        state = '已停止，請檢查錯誤'
    elif running and age is not None and age < 90:
        state = '執行中'
    elif running:
        state = '程序存在，心跳未更新，請檢查'
    elif full_finalize.get('status') == 'complete_local_and_publication_verified':
        state = '完整版與公開成果已完成'
    else:
        state = '程序未在執行，請檢查狀態'
    cool = ROOT / 'deliverables/cool_ready'
    report = cool / 'r14942154_report.pdf'
    archive = cool / 'r14942154.zip'
    ready = pipeline.get('cool_submission_ready') is True
    if not ready:
        # Local files can be complete even if a subsequent network upload fails.
        snapshot = read(cool / 'COOL_READY_MANIFEST.json')
        ready = (snapshot.get('status') == 'FORMAL_BASELINE_COOL_READY'
                 and report.is_file() and archive.is_file())
    lines = [
        'HW3 作業進度  |  更新時間 ' + now.strftime('%Y-%m-%d %H:%M:%S %Z'),
        '狀態：' + state,
        '目前階段：' + labels.get(stage, stage),
        '',
        f"音樂檢索：{'完成 100 / 100 題' if retrieval.get('status') == 'retrieval_complete' and retrieval.get('caption_count') == 100 else '進行中或待確認'}",
        f'必交音檔：{baseline} / 100 首',
        '正式評估：' + ('完成' if stages.get('required_metrics', {}).get('completed') is True else '尚未完成'),
        'NTU COOL 報告與 ZIP：' + ('已完成並保存' if ready else '尚未完成'),
        'GitHub 公開成果：' + ('正式版已發布並驗證' if destination.get('publishing_completed') is True
            else f"進度版已同步 {destination['progress_publication'].get('generated_clips', 0)} / 100 首"
            if destination.get('progress_publication', {}).get('remote_and_public_snapshot_verified') is True
            else '尚未完成'),
        '',
        '額外實驗：',
    ]
    for name, label in [('schedule_only', '排程'), ('rescale_only', '幅度修正'), ('combined', '組合')]:
        n, _ = count(name)
        lines.append(f'  {label}：{n} / 20 首')
    if ready:
        lines += ['', '必交檔案：', str(report), str(archive)]
    if pipeline.get('status') == 'stopped' or supervisor.get('status') == 'stopped':
        lines += ['', '錯誤：' + str(pipeline.get('error') or supervisor.get('error') or '請查看執行日誌')]
    lines += ['', '每 5 秒更新；關閉此視窗不會停止作業。請保持接電並開啟筆電。',
              '此視窗不會自動上傳 NTU COOL；請在檔案完成後自行繳交。']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        if sys.stdout.isatty() and not args.once:
            print('\033[2J\033[H', end='')
        print(render(), flush=True)
        if args.once:
            return
        time.sleep(5)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
