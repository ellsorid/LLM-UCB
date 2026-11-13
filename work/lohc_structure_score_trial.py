"""Shared scoring JSON utilities and DeepSeek client imports."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1] / "outputs"
sys.path.insert(0, str(ROOT / "LOHC_Agents_v2"))
from agents import DeepSeek, credential


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))
