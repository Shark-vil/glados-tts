#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import os
import sys
import time
import argparse
import inspect
import re
from functools import lru_cache
from typing import Optional, List, Tuple
from contextlib import ExitStack

import numpy as np
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel

MODEL_NAME = os.getenv("GLADOS_MODEL", "TeraTTS/glados2-g2p-vits")

ADD_TAIL_SEC = float(os.getenv("GLADOS_ADD_TAIL_SEC", "0.6"))

USE_RUACCENT = os.getenv("GLADOS_USE_RUACCENT", "1") == "1"
TOKENIZER_LOAD_DICT = os.getenv("GLADOS_LOAD_DICT", "1") == "1"

DEFAULT_SR = 22050
DEFAULT_LEN_SCALE = float(os.getenv("GLADOS_LENGTH_SCALE", "1.35"))

SENTENCE_SPLIT = os.getenv("GLADOS_SENTENCE_SPLIT", "1") == "1"
INTER_PAUSE_MS_DEFAULT = int(os.getenv("GLADOS_INTER_PAUSE_MS", "180"))
END_PAUSE_MS_DEFAULT = int(os.getenv("GLADOS_END_PAUSE_MS", "0"))

DOT_TO_ELLIPSIS = os.getenv("GLADOS_DOT_TO_ELLIPSIS", "0") == "1"

FFMPEG_AF_DEFAULT = os.getenv(
    "GLADOS_FFMPEG_AF",
    ""  # пример для лёгкого "гладос-окрас": acompressor=threshold=-20dB:ratio=3:attack=5:release=50,highpass=f=120,aecho=0.6:0.5:50:0.3
)

def _prepare_torch():
    try:
        import torch
        if not torch.cuda.is_available():
            torch.set_num_threads(max(1, int(os.getenv("GLADOS_TORCH_THREADS", "1"))))
        return torch
    except Exception:
        return None

@lru_cache(maxsize=1)
def get_accentizer():
    if not USE_RUACCENT:
        return None
    
    from ruaccent import RUAccent
    acc = RUAccent()  # без аргументов

    model_size = os.getenv("RUACCENT_MODEL_SIZE", "tiny2.1")  # tiny|tiny2|tiny2.1|turbo2|turbo3|turbo3.1|turbo|big_poetry
    use_dict   = os.getenv("RUACCENT_USE_DICT", "1") == "1"
    device     = os.getenv("RUACCENT_DEVICE", "CUDA")          # "CPU" или "CUDA" (нужен onnxruntime-gpu)
    workdir    = os.getenv("RUACCENT_WORKDIR", "./ruaccent_cache")
    tiny_mode  = os.getenv("RUACCENT_TINY_MODE", "0") == "1"  # True отключает часть пайплайна и словарь

    os.makedirs(workdir, exist_ok=True)

    # правильный вызов: путь — через load(...)
    acc.load(
        omograph_model_size=model_size,
        use_dictionary=use_dict,
        device=device,
        workdir=workdir,
        tiny_mode=tiny_mode
    )

    return acc

def _apply_fade(audio: np.ndarray, sr: int, fade_ms: int = 5) -> np.ndarray:
    n = max(1, int(sr * fade_ms / 1000.0))
    if n * 2 >= len(audio):
        return audio
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = fade_in[::-1]
    out = audio.copy()
    out[:n] *= fade_in
    out[-n:] *= fade_out
    return out

def _zeros(duration_ms: int, sr: int) -> np.ndarray:
    if duration_ms <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.zeros(int(sr * duration_ms / 1000.0), dtype=np.float32)

def float_to_opus_bytes(audio: np.ndarray, sr: int, af_override: Optional[str] = None) -> bytes:
    a = np.asarray(audio, dtype=np.float32).flatten()
    a = np.clip(a, -1.0, 1.0)
    i16 = (a * 32767.0).astype(np.int16)
    import shutil, subprocess
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found")
    af = (af_override if (af_override is not None and af_override != "") else FFMPEG_AF_DEFAULT)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "s16le",
        "-ar", str(int(sr)),
        "-ac", "1",
        "-i", "pipe:0",
    ]
    if af:
        cmd += ["-af", af]
    cmd += [
        "-c:a", "libopus",
        "-b:a", "64k",
        "-f", "ogg",
        "pipe:1",
    ]
    try:
        p = subprocess.run(
            cmd,
            input=i16.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return p.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg encode failed: {e.stderr.decode(errors='ignore')}")

def _normalize_punct(s: str) -> str:
    t = s
    t = re.sub(r"\*[^*\n]+\*", " ", t)  # убрать *курсивные ремарки*
    t = t.replace(" - ", " — ")
    t = re.sub(r"\s*—\s*", " — ", t)
    t = re.sub(r"\.{2,}", "…", t)
    if DOT_TO_ELLIPSIS:
        t = re.sub(r"(?<![!?…])\.", "…", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    if not t.endswith((".", "…", "!", "?")):
        t += "."
    return t

_SENT_PAT = re.compile(
    r"""
    [^.!?…]+[.!?…]      # кусок, заканчивающийся на конечный знак
    |                   # или
    [^.!?…]+$           # хвост без конечного знака
    """,
    re.VERBOSE | re.UNICODE
)

def _split_sentences(t: str) -> List[str]:
    parts = [p.strip() for p in _SENT_PAT.findall(t) if p and p.strip()]
    # Доп. разбиение по " — " и ";", чтобы добавить микропауз
    out: List[str] = []
    for p in parts:
        chunks = re.split(r"\s+[—;]\s+", p)
        for i, c in enumerate(chunks):
            c = c.strip()
            if not c:
                continue
            if i < len(chunks) - 1 and not re.search(r"[.!?…]$", c):
                c += "…"  # мягкий конец фразы
            out.append(c)
    return out if out else [t]

class TTSPipe:
    def __init__(self,
                 model_name: str = MODEL_NAME,
                 add_tail_sec: float = ADD_TAIL_SEC,
                 tokenizer_load_dict: bool = TOKENIZER_LOAD_DICT,
                 default_len_scale: float = DEFAULT_LEN_SCALE):
        self.torch = _prepare_torch()
        from TeraTTS import TTS
        self.tts = TTS(
            model_name,
            add_time_to_end=add_tail_sec,
            tokenizer_load_dict=tokenizer_load_dict
        )
        self.sample_rate = int(getattr(self.tts, "sample_rate", DEFAULT_SR))
        self.accentizer = get_accentizer()
        self.default_len_scale = default_len_scale
        self._supports_len_scale = False
        try:
            sig = inspect.signature(self.tts.__call__)
            self._supports_len_scale = ("length_scale" in sig.parameters)
        except Exception:
            self._supports_len_scale = False

    def _preprocess(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return t
        if self.accentizer:
            try:
                t = self.accentizer.process_all(t)
            except Exception:
                pass
        return t

    def _tts_call(self, txt: str, length_scale: Optional[float]) -> np.ndarray:
        ls = float(length_scale) if (length_scale is not None) else self.default_len_scale
        with ExitStack() as stack:
            if self.torch is not None:
                if hasattr(self.torch, "inference_mode"):
                    stack.enter_context(self.torch.inference_mode())
                else:
                    stack.enter_context(self.torch.no_grad())
            try:
                if self._supports_len_scale:
                    audio = self.tts(txt, length_scale=ls)
                else:
                    audio = self.tts(txt)
            except TypeError:
                audio = self.tts(txt)
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        return np.asarray(audio, dtype=np.float32).flatten()

    def synth(
        self,
        text: str,
        length_scale: Optional[float] = None,
        use_accent: Optional[bool] = None,
        normalize_punct: bool = True,
        inter_pause_ms: Optional[int] = None,
        end_pause_ms: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        t = text.strip()
        if normalize_punct:
            t = _normalize_punct(t)
        processed = t if (use_accent is False) else self._preprocess(t)

        if SENTENCE_SPLIT:
            sents = _split_sentences(processed)
        else:
            sents = [processed]

        ip_ms = INTER_PAUSE_MS_DEFAULT if inter_pause_ms is None else max(0, int(inter_pause_ms))
        ep_ms = END_PAUSE_MS_DEFAULT if end_pause_ms is None else max(0, int(end_pause_ms))

        chunks: List[np.ndarray] = []
        for i, s in enumerate(sents):
            a = self._tts_call(s, length_scale)
            a = _apply_fade(a, self.sample_rate, 5)
            chunks.append(a)
            if i != len(sents) - 1 and ip_ms > 0:
                chunks.append(_zeros(ip_ms, self.sample_rate))
        if ep_ms > 0:
            chunks.append(_zeros(ep_ms, self.sample_rate))

        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, self.sample_rate

app = FastAPI(title="GLaDOS TTS (TeraTTS)")

class SynthesizeIn(BaseModel):
    text: str
    filename: Optional[str] = None
    length_scale: Optional[float] = None
    use_accent: Optional[bool] = None
    normalize_punct: Optional[bool] = True
    inter_pause_ms: Optional[int] = None
    end_pause_ms: Optional[int] = None
    ffmpeg_af: Optional[str] = None

@app.get("/health")
def health():
    pipe = get_tts_pipe()
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "sr": pipe.sample_rate,
        "ruaccent": bool(pipe.accentizer),
        "len_scale_supported": pipe._supports_len_scale,
        "defaults": {
            "length_scale": pipe.default_len_scale,
            "inter_pause_ms": INTER_PAUSE_MS_DEFAULT,
            "end_pause_ms": END_PAUSE_MS_DEFAULT,
            "ffmpeg_af": FFMPEG_AF_DEFAULT,
        }
    }

@app.post("/synthesize", responses={200: {"content": {"audio/ogg": {}}}})
def synthesize(inp: SynthesizeIn):
    if not inp.text or not inp.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    t0 = time.time()
    try:
        audio, sr = get_tts_pipe().synth(
            inp.text,
            length_scale=inp.length_scale,
            use_accent=inp.use_accent,
            normalize_punct=bool(inp.normalize_punct),
            inter_pause_ms=inp.inter_pause_ms,
            end_pause_ms=inp.end_pause_ms,
        )
        data = float_to_opus_bytes(audio, sr, af_override=inp.ffmpeg_af)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"synthesis error: {e}")
    headers = {"X-Elapsed": f"{(time.time()-t0):.3f}s"}
    if inp.filename:
        name = inp.filename
        if not name.lower().endswith(".ogg"):
            name += ".ogg"
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return Response(content=data, media_type="audio/ogg", headers=headers)

@lru_cache(maxsize=1)
def get_tts_pipe() -> TTSPipe:
    return TTSPipe(MODEL_NAME, ADD_TAIL_SEC, TOKENIZER_LOAD_DICT, DEFAULT_LEN_SCALE)

def run_cli(text_arg: Optional[str]):
    pipe = get_tts_pipe()
    def save_one(txt: str, out_name: Optional[str] = None):
        audio, sr = pipe.synth(txt)
        data = float_to_opus_bytes(audio, sr)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        fn = out_name or f"glados_{stamp}.ogg"
        with open(fn, "wb") as f:
            f.write(data)
        print(f"[ok] saved: {fn}")
    if text_arg is not None:
        save_one(text_arg)
        return
    print("GLaDOS dev-CLI. Печатай текст, пустая строка — выход.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            break
        save_one(line.strip())

def main():
    p = argparse.ArgumentParser(description="GLaDOS TTS API / CLI")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cli", nargs="?", const="", help="CLI режим: текст или пусто (интерактив)")
    args = p.parse_args()
    if args.cli is not None:
        text_arg = args.cli if args.cli != "" else None
        return run_cli(text_arg)
    import uvicorn
    pipe = get_tts_pipe()
    print(f"[GLaDOS] model={MODEL_NAME} sr={pipe.sample_rate} "
          f"ruaccent={bool(pipe.accentizer)} "
          f"len_scale_supported={pipe._supports_len_scale} "
          f"default_len_scale={pipe.default_len_scale} "
          f"inter_pause_ms={INTER_PAUSE_MS_DEFAULT} end_pause_ms={END_PAUSE_MS_DEFAULT}")
    uvicorn.run("glados_api:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
