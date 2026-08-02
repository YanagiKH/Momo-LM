from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path


class OfflineSpeech:
    def synthesize(self, text: str, output: Path, *, rate: int = 170) -> dict[str, str]:
        if not text.strip():
            raise ValueError("Speech text is empty")
        output.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            self._windows_sapi(text, output, rate)
            return {"path": str(output), "engine": "Windows SAPI"}
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable:
            subprocess.run(
                [executable, "-s", str(max(80, min(rate, 350))), "-w", str(output), text],
                check=True,
                timeout=60,
                capture_output=True,
            )
            return {"path": str(output), "engine": Path(executable).name}
        self._fallback_tones(text, output)
        return {"path": str(output), "engine": "Momo fallback tones"}

    @staticmethod
    def _windows_sapi(text: str, output: Path, rate: int) -> None:
        escaped_text = text.replace("'", "''")
        escaped_path = str(output.resolve()).replace("'", "''")
        sapi_rate = max(-10, min(10, round((rate - 170) / 15)))
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate={sapi_rate}; $s.SetOutputToWaveFile('{escaped_path}'); "
            f"$s.Speak('{escaped_text}'); $s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=60)

    @staticmethod
    def _fallback_tones(text: str, output: Path) -> None:
        sample_rate = 22_050
        frames = bytearray()
        for character in text[:240]:
            frequency = 180 + (ord(character) % 24) * 12
            duration = 0.055 if not character.isspace() else 0.035
            for index in range(int(sample_rate * duration)):
                envelope = min(1.0, index / 90) * min(1.0, (sample_rate * duration - index) / 90)
                value = int(8000 * envelope * math.sin(2 * math.pi * frequency * index / sample_rate))
                frames.extend(struct.pack("<h", value))
        with wave.open(os.fspath(output), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sample_rate)
            stream.writeframes(frames)
