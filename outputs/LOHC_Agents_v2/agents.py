"""Run bounded DeepSeek agents against the Cell 17 prototype and native GA pilot."""

import argparse
import hashlib
import http.client
import json
import os
import re
import time
import io_helpers as demo

KEY = re.compile(r"(?<![\w-])sk-[A-Za-z0-9_-]{16,}")
def credential(path):
    if path is None:
        return os.environ["LOHC_LLM_API_KEY"]
    data = path.read_bytes()
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    lines = data.decode(encoding).splitlines()
    matches = set(KEY.findall("\n".join(s for s in lines if "deepseek" in s.lower())))
    matches = matches or set(KEY.findall("\n".join(lines)))
    if len(matches) != 1:
        raise ValueError("Expected one unambiguous DeepSeek credential")
    return matches.pop()


class DeepSeek:
    def __init__(self, key, output):
        self.key = key
        self.output = output

    def run(self, role, instruction, packet):
        payload = {
            "model": "deepseek-flash", "thinking": {"type": "disabled"},
            "temperature": 0, "max_tokens": 1100, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": instruction},
                         {"role": "user", "content": demo.json_text(packet)}],
        }
        data = demo.json_text(payload).encode("utf-8")
        conn = http.client.HTTPSConnection("api.deepseek.com", timeout=65)
        start = time.perf_counter()
        status, result = None, {}
        try:
            conn.request("POST", "/chat/completions", body=data,
                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
            response = conn.getresponse()
            status = response.status
            if status != 200:
                raise RuntimeError(f"DeepSeek HTTP {status} in {role}")
            result = json.loads(response.read().decode("utf-8").replace(self.key, "[redacted]"))
        finally:
            conn.close()
            audit = {"role": role, "http_status": status, "elapsed_seconds": round(time.perf_counter()-start, 3),
                     "request_sha256": hashlib.sha256(data).hexdigest(), "model": result.get("model"),
                     "response_id": result.get("id"), "usage": result.get("usage")}
            demo.write_json(self.output / f"{role}_audit.json", audit)
        choice = result["choices"][0]
        content = KEY.sub("[redacted]", choice["message"].get("content") or "")
        (self.output / f"{role}_response.txt").write_text(content, encoding="utf-8")
        if choice.get("finish_reason") == "length" or not content.strip():
            raise ValueError(f"Empty or truncated response in {role}")
        return json.loads(content)
