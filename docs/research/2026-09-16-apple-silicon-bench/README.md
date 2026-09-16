# M5 Pro 本地 TTS 实测复现包（2026-09-16）

机器：MacBook Pro Mac17,9 / Apple M5 Pro / 24 GB / macOS 26
参考音频：`ref.wav`（9.47 s，22.05 kHz mono，`say -v Tingting` + `ffmpeg -ar 22050 -ac 1`）

| 脚本 | 引擎 | 运行命令 |
|---|---|---|
| `bench.py` | IndexTTS-2.5 (`index-tts-2.5-mlx`) | `uv run --with index-tts-2.5-mlx python bench.py` |
| `cbbench.py` | Chatterbox V3 (mlx-audio) | `HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_XET=1 uv run --with mlx-audio --with soundfile --with numpy --python 3.12 python cbbench.py` |
| `qwenbench.py` | Qwen3-TTS 0.6B Base 8bit (mlx-audio) | 同上（换脚本） |
| `kbench.py` | Kokoro-82M-v1.1-zh (kokoro-onnx) | 需先下 `kokoro-v1.1-zh.onnx` + `voices-v1.1-zh.bin` + `config.json`；`uv run --with kokoro-onnx --with "misaki-fork[zh]" --with soundfile --python 3.12 python kbench.py` |
| `asr.py` / `cbasr.py` / `qwasr.py` | ASR 复核 | `uv run --with faster-whisper --python 3.12 python <script>` |

注意：本机全局 `HF_ENDPOINT=https://hf-mirror.com`，该镜像当日多次 SSL 握手失败；跑 mlx-audio 时建议显式覆盖为 `https://huggingface.co`。
