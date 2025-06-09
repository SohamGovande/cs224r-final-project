#!/usr/bin/env python3
# collect_pairs.py – generate (good, bad) completion pairs for every HumanEval-VL task

import argparse
import ast
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from typing import List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info

# ───────────────────────── embedded reward helpers ──────────────────────────
try:
    import signal
    _HAS_SIGNAL = True
except ImportError:
    _HAS_SIGNAL = False


class _PythonCodeExecutor:
    """
    Executes a candidate reply against the HumanEval unit tests.

    • Mode 1  – reply is treated as *body only* and indented below the function header.
    • Mode 2  – reply is pasted verbatim (allows normal `def …` solutions).

    If either mode passes all tests → reward = 1.0, else 0.0.
    """

    # ---------- helpers -----------------------------------------------------
    @staticmethod
    def extract_code(text: str) -> str:
        blocks = (re.findall(r"```python\n(.*?)```", text, re.S) or
                  re.findall(r"```\n(.*?)```", text, re.S))
        return (blocks[-1] if blocks else text).strip()

    @staticmethod
    def _syntax_ok(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    @staticmethod
    @contextmanager
    def _timeout(sec: int):
        if _HAS_SIGNAL and hasattr(signal, "SIGALRM"):
            def handler(sig, frame): raise TimeoutError
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(sec)
            try:
                yield
            finally:
                signal.alarm(0)
        else:
            yield

    # ---------- public ------------------------------------------------------
    @classmethod
    def execution_reward(cls, reply: str, *, prompt: str,
                         test_code: str, entry_point: str,
                         timeout: int = 5) -> float:
        code = cls.extract_code(reply)
        if not cls._syntax_ok(code):
            return 0.0

        def make_harness(indent: bool) -> str:
            if indent:
                body = textwrap.indent("# MODEL COMPLETION\n" + code.rstrip(), "    ")
            else:
                body = "# MODEL COMPLETION\n" + code.rstrip()
            return (
                f"# PROMPT\n{prompt}\n"
                f"{body}\n\n"
                f"{test_code}\n\n"
                f"if __name__ == '__main__':\n"
                f"    check({entry_point})\n"
            )

        for indent_mode in (True, False):
            harness = make_harness(indent_mode)
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(harness)
                path = tmp.name
            try:
                with cls._timeout(timeout):
                    rc = subprocess.run([sys.executable, path],
                                        capture_output=True).returncode
            except TimeoutError:
                rc = 1
            finally:
                os.unlink(path)

            if rc == 0:          # tests passed in this mode
                return 1.0

        return 0.0


# ───────────────────────── generation parameters ────────────────────────────
NUM_SAMPLES   = 10
MAX_NEW       = 512
TEMP          = 0.7
TOP_P         = 0.95
MAX_ATTEMPTS  = 24
SEED          = 42


# ─────────────────────────────── helper ------------------------------------
def _is_good(reply: str, rec) -> bool:
    """Good iff execution reward == 1.0."""
    return _PythonCodeExecutor.execution_reward(
        reply,
        prompt      = rec["prompt"],
        test_code   = rec["test"],
        entry_point = rec["entry_point"],
    ) == 1.0


# ─────────────────────────────── main logic ────────────────────────────────
def main(out_path: pathlib.Path, data_path: pathlib.Path) -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading Qwen/Qwen2.5-VL-3B-Instruct …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    records: List[dict] = [json.loads(l) for l in data_path.open()]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        for idx, rec in enumerate(tqdm(records, desc="HumanEval-VL")):
            img = Image.open(rec["image"]).convert("RGB")

            # chat template
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": rec["prompt"]},
                ],
            }]
            chat = proc.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            base_in = proc(text=[chat], images=img_inputs, videos=vid_inputs,
                           padding=True, return_tensors="pt").to(model.device)

            good, bad, seen = [], [], set()
            attempts = 0
            while (not good or not bad) and attempts < MAX_ATTEMPTS:
                attempts += 1
                outs = model.generate(
                    **base_in, do_sample=True, top_p=TOP_P, temperature=TEMP,
                    num_return_sequences=NUM_SAMPLES, max_new_tokens=MAX_NEW
                )
                for o in outs:
                    reply = proc.decode(o[len(base_in.input_ids[0]):],
                                        skip_special_tokens=True).strip()
                    if reply in seen:
                        continue
                    seen.add(reply)
                    if _is_good(reply, rec):
                        good.append(reply)
                    else:
                        bad.append(reply)

                tqdm.write(f"[task {idx}] attempt {attempts}: "
                           f"{len(good)} good • {len(bad)} bad")

            if not good or not bad:
                tqdm.write(f"[task {idx}] ⚠️  unable to get both good & bad; skipping")
                continue

            g, b = good[0], bad[0]                       # keep names handy
            fout.write(json.dumps({
                "task_id":   rec.get("task_id", rec.get("id", idx)),
                "good":      g,
                "bad":       b,
                "good_code": _PythonCodeExecutor.extract_code(g),
                "bad_code":  _PythonCodeExecutor.extract_code(b),
            }) + "\n")

    print(f"✓ Done – pairs written to {out_path.resolve()}")


# ───────────────────────────── command line ────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path,
                    default=pathlib.Path("dataset_with_testcode_20250601_092346.jsonl"),
                    help="JSONL with HumanEval-VL items")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("humaneval_pairs_vl/good_bad_pairs.jsonl"),
                    help="Destination JSONL for good/bad pairs")
    args = ap.parse_args()
    main(out_path=args.out, data_path=args.data)
