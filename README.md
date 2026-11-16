# HW3：製作進度（尚非正式繳交版）

學號：r14942154。更新：2026-11-16T04:56:01.474011+00:00。

已完成 **66 / 100 首** baseline 音樂；100 題 CLAP 檢索已完成。

此處先公開程式、模型固定來源、檢索結果與已完成的 MP3。正式報告與 COOL ZIP 會在全部必要生成、評估與格式驗證後更新。這份進度快照不能作為完整作業繳交。

`PROGRESS_SNAPSHOT.json` 是該次上傳的歷史快照。生成在本機持續進行，每新增約 5 首或有新的執行階段時分批更新。

音檔：`results/baseline/generated_samples/`；檢索：`results/retrieval/`；固定設定：`analysis/experiment_protocol_frozen.json`。

環境使用 Python 3.11，版本見 requirements 與 lock 檔。原始課程 caption CSV、MusicCaps 資料集、模型權重與登入快取均未公開。重現時須自行備妥原始課程資料，並先親自接受官方模型條款；使用 `reproduce.py` 與原始固定設定重現時，需等完整 100 首的正式 manifest 完成。

SAO/CLAP 依固定 HF LFS 完整雜湊驗證；VGGish/PaSST 使用固定鏡像 SHA 與官方 release 大小核對。模型與程式授權資訊見各來源及 `licenses/`。
