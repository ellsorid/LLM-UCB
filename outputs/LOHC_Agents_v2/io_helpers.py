"""Shared JSON and hashing helpers."""

import hashlib
import json


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def write_json(path, value):
    path.write_text(json_text(value) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
