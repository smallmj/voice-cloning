"""Loop runner: mimics the real sidecar process tree during generation.

The qwen3-tts-mlx engine spawns its worker with subprocess.Popen from inside
the sidecar python process (engines/qwen3_tts.py). During a long generation,
app quit leaves that worker behind if only the uv wrapper is signalled.
Here a persistent `sleep` stands in for the engine worker (same tree shape:
uv -> python(sidecar) -> worker), then the real sidecar serves as usual.
"""
import runpy
import subprocess
import sys

subprocess.Popen(["sleep", "300"])
sys.argv = ["voiceclone_sidecar"] + sys.argv[1:]
runpy.run_module("voiceclone_sidecar", run_name="__main__")
