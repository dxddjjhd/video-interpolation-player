# 视频实时插帧播放器

本地视频播放器，在播放过程中**实时生成中间帧**，把视频帧率提升到 2x/4x，
画面更流畅（适合 60fps 视频插到 120fps/240fps 观看）。

## 功能

- 🎬 实时插帧：2x / 4x 可切换（2x = 60fps→120fps）
- 🔍 **超分辨率放大**：关闭 / Lanczos 2x/4x / AI 2x/4x（RealESRGAN）
  - AI 超分自动守护：每帧耗时超过显示预算时自动切换 Lanczos，播放不卡
  - RealESRGAN 模型仅 ~6.5MB，首次选择 AI 超分时自动下载
- 🤖 双插帧引擎，可随时切换：
  - **RIFE (AI)**：深度学习光流插帧，画质最好，需要 NVIDIA 显卡 (CUDA)
  - **OpenCV 光流**：传统算法，纯 CPU 也能跑（兼容降级）
- 🔄 自动引擎选择：启动时检测 CUDA；打开视频时**实测插帧速度**，
  RIFE 达不到 2x 实时预算会自动切换光流引擎（状态栏会提示原因）
- 🎚 插帧分辨率缩放（原尺寸/75%/50%）：RIFE 在降分辨率下插帧、
  显示时放大回原尺寸，弱显卡也能保住实时
- 🎧 音画同步：以音频时钟为基准；音频饥饿时自动切换墙钟模式
- 🛟 实时降级：插帧跟不上时自动跳过中间帧出原帧，保证播放不卡
- 🎛 播放控制：播放/暂停、进度条拖拽 seek、←/→ 快退快进 5s、拖放文件打开

## 安装

```bash
pip install -r requirements.txt
```

- RIFE 引擎需要 NVIDIA 驱动 + CUDA 运行库（onnxruntime-gpu 自带依赖检查）
- 首次启动 RIFE 时自动下载模型包（约 190MB，来自 vs-mlrt 官方模型发布），
  也可以手动放任意 `rife*.onnx` 到 `player/interpolate/models/` 跳过下载

## 使用

```bash
python main.py                 # 启动后点"打开"选视频
python main.py 视频.mp4        # 直接播放
```

快捷键：`空格` 播放/暂停，`←`/`→` 快退/快进 5s。

界面右下角状态栏实时显示：当前引擎、生成帧率（应 ≈ 源帧率 × 倍率）、
是否处于降级状态。

## 测试

```bash
python tools/make_test_video.py   # 生成 60fps 合成测试视频（含音频）
python -m pytest tests/ -v
```

## 架构

```
解码线程 ──有界队列──▶ 插帧线程 ──按时钟节拍──▶ 超分(可选) ──▶ QPainter 渲染
  │  (PyAV/FFmpeg)     │ (RIFE ONNX 或光流)       │ (Lanczos/AI)
  ▼                    ▼
音频写入(QAudioSink)   音频时钟 = A/V 同步基准
```

- `player/core/decoder.py` — PyAV 双 container 解码（视频/音频各一个，互不干扰）
- `player/core/player_engine.py` — 三级流水线 + 时钟同步 + 降级
- `player/interpolate/rife.py` — RIFE ONNX 推理（CUDA 优先）
- `player/interpolate/optical_flow.py` — Farneback 光流插帧
- `player/superres/real_esrgan.py` — RealESRGAN ONNX 超分（CUDA 优先）
- `player/superres/lanczos.py` — Lanczos 超分（CPU 实时兜底）

## 已知限制

- RIFE 实时 1080p 2x 需要 RTX 3060 及以上级别显卡；4x 建议 RTX 4070+ 或降低分辨率
- 虚拟化/共享 GPU（云主机常见）算力受限：RIFE 可能达不到实时，
  播放器会自动切换光流引擎或降分辨率插帧（状态栏可见提示）
- 光流引擎在快速运动/遮挡场景可能有伪影
