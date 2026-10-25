"""Completion engines.

An engine turns a transcript into a stream of `(text, token_id)` pieces.
The transcript arrives as segments `(text, token_ids)`; the last segment is
where generation starts. Engines keep whatever cache they can across calls,
so a call that shares a prefix with the previous one only pays for the
difference, whether that's the next turn or a different branch of the tree.

- `MLXEngine` runs a model in-process on Apple silicon and owns the decode
  loop. Token ids are reported so the session can keep the model's own
  tokens in the transcript instead of re-encoding its output.
- `RemoteEngine` streams from an OpenAI-compatible `/v1/completions` server
  (vLLM, SGLang, llama.cpp's llama-server, mlx_lm.server). It reports text
  only; the server's prefix cache does the reuse.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

Segment = tuple[str, "list[int] | None"]
Piece = tuple[str, "int | None"]


class Engine(Protocol):
    tokenized: bool  # True if encode/decode work and pieces carry token ids

    def encode(self, text: str) -> list[int] | None: ...

    def decode(self, tokens: list[int]) -> str: ...

    def warm(self, segments: Sequence[Segment]) -> None: ...

    def stream(self, segments: Sequence[Segment], max_tokens: int,
               interactive: bool = False) -> Iterator[Piece]:
        """`interactive` marks a stream read while the user is typing (the
        ghost guess), which must never stall the prompt for long."""
        ...


class _Detok:
    """Incremental detokenizer. Each token is decoded together with an
    anchor token to its left, so tokenizers that strip a leading space at the
    start of a decode (SentencePiece) still report it, and the anchor moves
    forward so a decode never spans more than a short window."""

    WINDOW = 32

    def __init__(self, decode: Any, anchor: list[int]):
        self.decode = decode
        self._anchor(anchor)

    def _anchor(self, ids: list[int]) -> None:
        self.ids = list(ids)
        self.skip = len(self.decode(self.ids))
        self.emitted = 0

    def add(self, tok: int) -> str:
        self.ids.append(tok)
        text = self.decode(self.ids)[self.skip:]
        if text.endswith("�"):  # incomplete UTF-8; wait for the rest
            return ""
        new = text[self.emitted:]
        self.emitted = len(text)
        if len(self.ids) > self.WINDOW:
            self._anchor(self.ids[-1:])
        return new

    def flush(self) -> str:
        return self.decode(self.ids)[self.skip:][self.emitted:]


def _lcp(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    i = 0
    while a[i] == b[i]:
        i += 1
    return i


def _cached(model_id: str) -> str:
    """The local snapshot of an already-downloaded HF repo, so a cached model
    loads without download progress bars; otherwise `model_id` unchanged."""
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download
        from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

        return snapshot_download(model_id, local_files_only=True, allow_patterns=DEFAULT_ALLOW_PATTERNS)
    except Exception:
        return model_id


class MLXEngine:
    """In-process MLX model with a cache that rewinds.

    `self.tokens` is exactly what the cache has consumed. Moving to a new
    transcript rewinds to the longest common prefix and prefills the rest.
    Attention caches rewind by trimming. Recurrent layers (Qwen3.5's
    linear-attention layers) can't trim, so their state is snapshotted at
    turn boundaries and at line starts while decoding; MLX arrays are
    immutable values, so a snapshot is a list of references, not a copy.
    """

    tokenized = True
    PREFILL_STEP = 2048
    MAX_SNAPSHOTS = 16

    def __init__(self, model_id: str, temperature: float, top_p: float, bits: int = 4):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step, generation_stream
        from mlx_lm.models.cache import ArraysCache, make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.utils import quantize_model

        self._mx = mx
        self._generate_step: Any = generate_step
        self._stream = generation_stream
        self._ArraysCache = ArraysCache
        self._make_cache = make_prompt_cache
        loaded: Any = load(_cached(model_id), lazy=True, return_config=True)
        model, self.tok, config = loaded
        # Decoding is memory-bound, so quantizing the weights at load (under a
        # second) roughly triples speed. Checkpoints that are already
        # quantized load as they are.
        if bits < 16 and "quantization" not in config:
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = quantize_model(model, config, 64, bits)
        mx.eval(model.parameters())
        self.model: Any = model
        self.sampler = make_sampler(temp=temperature, top_p=top_p if top_p < 1.0 else 0.0)
        self.eos = set(self.tok.eos_token_ids)
        # A terminal never prints chat or tool markup, and sampling one of
        # those tokens (<think>, <|im_start|>, ...) is how a "base" model
        # trained on instruction data slips into assistant mode. Only EOS
        # stays reachable.
        banned = [i for i in getattr(self.tok, "added_tokens_decoder", {}) if i not in self.eos]
        self._processors: list[Any] = [self._ban(mx.array(banned))] if banned else []
        # Special tokens the tokenizer puts in front of a document (BOS on
        # Gemma/Llama, nothing on Qwen); prepended once to every transcript.
        with_special = self.tok.encode("x")
        self.bos = with_special[: len(with_special) - len(self.tok.encode("x", add_special_tokens=False))]
        self._reset()
        self.recurrent = any(isinstance(c, ArraysCache) for c in self.cache)

    def _ban(self, ids: Any) -> Any:
        mx = self._mx

        def processor(tokens: Any, logits: Any) -> Any:
            logits[:, ids] = -mx.inf
            return logits
        return processor

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False) if text else []

    def decode(self, tokens: list[int]) -> str:
        return self.tok.decode(tokens, skip_special_tokens=True)

    # -- cache bookkeeping --------------------------------------------------

    def _reset(self) -> None:
        # Bookkeeping is emptied before the new cache goes in, so an interrupt
        # partway through leaves fewer listed tokens than cached, never more;
        # _recover handles that direction.
        cache = self._make_cache(self.model)
        self.tokens: list[int] = []
        self.snaps: list[tuple[int, list[Any]]] = []  # turn boundaries, ascending
        self.line_snap: tuple[int, list[Any]] | None = None
        self.cache: list[Any] = cache

    def _snapshot(self) -> tuple[int, list[Any]]:
        return len(self.tokens), [list(c.cache) if isinstance(c, self._ArraysCache) else None
                                  for c in self.cache]

    def _mark_boundary(self) -> None:
        if self.recurrent and (not self.snaps or self.snaps[-1][0] < len(self.tokens)):
            self.snaps.append(self._snapshot())
            del self.snaps[:-self.MAX_SNAPSHOTS]

    def _restorable(self) -> bool:
        return all(isinstance(c, self._ArraysCache) or c.is_trimmable() for c in self.cache)

    def _restore(self, pos: int, arrays: list[Any] | None) -> None:
        for i, c in enumerate(self.cache):
            if arrays is not None and arrays[i] is not None:
                c.cache = list(arrays[i])
            else:
                c.trim(c.size() - pos)
        del self.tokens[pos:]

    def _rewind(self, n: int) -> None:
        """Make the cache cover exactly `self.tokens[:n]`."""
        if n >= len(self.tokens):
            return
        old = list(self.tokens)
        self.snaps = [s for s in self.snaps if s[0] <= n]
        if self.line_snap is not None and self.line_snap[0] > n:
            self.line_snap = None
        if not self.recurrent and self._restorable():
            self._restore(n, None)
            return
        best = max(self.snaps + ([self.line_snap] if self.line_snap else []),
                   key=lambda s: s[0], default=None)
        if best is not None and self._restorable():
            self._restore(*best)
        else:
            self._reset()
        self._prefill(old[len(self.tokens):n])

    def _prefill(self, tokens: list[int]) -> None:
        mx = self._mx
        for i in range(0, len(tokens), self.PREFILL_STEP):
            chunk = tokens[i:i + self.PREFILL_STEP]
            with mx.stream(self._stream):
                self.model(mx.array(chunk)[None], cache=self.cache)
                mx.eval([c.state for c in self.cache])
            self.tokens.extend(chunk)

    def _recover(self) -> None:
        """After an interrupted model call some layers may have consumed
        tokens that `self.tokens` doesn't list. Put every layer back on a
        known position, or start over if that isn't possible."""
        n = len(self.tokens)
        self.line_snap = None
        self.snaps = [s for s in self.snaps if s[0] <= n]
        target = (n, None) if not self.recurrent else (self.snaps[-1] if self.snaps else None)
        trimmed = [c for c in self.cache if not isinstance(c, self._ArraysCache)]
        if (target is None or not self._restorable()
                or any(c.size() < target[0] for c in trimmed)):  # a restore may only shrink
            self._reset()
        else:
            self._restore(*target)

    def _sync(self, segments: Sequence[Segment]) -> tuple[list[int], int]:
        """Rewind/prefill so the cache covers everything but the last
        segment, and return (full target, its boundary)."""
        ctx = list(self.bos)
        for text, toks in segments[:-1]:
            ctx += toks if toks is not None else self.encode(text)
        last = segments[-1]
        target = ctx + (last[1] if last[1] is not None else self.encode(last[0]))
        # Leave at least one token unconsumed, wherever the segments split:
        # sampling needs its logits.
        keep = max(0, len(target) - 1)
        try:
            self._rewind(min(_lcp(self.tokens, target), keep))
            if len(self.tokens) < min(len(ctx), keep):
                self._prefill(target[len(self.tokens):min(len(ctx), keep)])
            if len(self.tokens) == len(ctx):
                self._mark_boundary()
        except BaseException:
            self._recover()
            raise
        return target, len(ctx)

    # -- public -------------------------------------------------------------

    def warm(self, segments: Sequence[Segment]) -> None:
        target, _ = self._sync(list(segments) + [("", [])])
        try:
            self._prefill(target[len(self.tokens):])
        except BaseException:
            self._recover()
            raise
        self._mark_boundary()

    def stream(self, segments: Sequence[Segment], max_tokens: int,
               interactive: bool = False) -> Iterator[Piece]:
        target, _ = self._sync(segments)
        rest = target[len(self.tokens):]
        if not rest:
            return  # an empty transcript has nothing to condition on
        detok = _Detok(self.decode, rest[-1:])
        steps = self._generate_step(self._mx.array(rest), self.model, max_tokens=-1,
                                    sampler=self.sampler, logits_processors=self._processors,
                                    prompt_cache=self.cache, prefill_step_size=self.PREFILL_STEP)
        count = 0
        try:
            # generate_step feeds each token back into the cache before
            # yielding it, so after a yield the cache has consumed it.
            for tok, _ in steps:
                if count == 0:
                    self.tokens.extend(rest)
                self.tokens.append(tok)
                if tok in self.eos:
                    break
                text = detok.add(tok)
                if "\n" in text and self.recurrent:
                    self.line_snap = self._snapshot()
                yield text, tok
                count += 1
                if count >= max_tokens:
                    break
            tail = detok.flush()
            if tail:
                yield tail, None
        except GeneratorExit:
            raise
        except BaseException:
            self._recover()
            raise
        finally:
            steps.close()


class RemoteEngine:
    """Streams raw completions from an OpenAI-compatible server."""

    tokenized = False
    # An interactive stream (read while the user types) must never hold the
    # prompt: each read gives up after INTERACTIVE_READ seconds of silence,
    # and the whole stream ends INTERACTIVE_DEADLINE seconds after it began.
    INTERACTIVE_READ = 1.0
    INTERACTIVE_DEADLINE = 5.0

    def __init__(self, url: str, model: str | None, temperature: float, top_p: float,
                 api_key: str | None = None):
        base = url.rstrip("/")
        self.base = base if base.endswith("/v1") else base + "/v1"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.params = {"temperature": temperature, "top_p": top_p}
        self.model = model or self._default_model()

    def _default_model(self) -> str:
        req = urllib.request.Request(self.base + "/models", headers=self.headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)["data"]
        if not data:
            raise RuntimeError(f"{self.base}/models lists no models")
        return data[0]["id"]

    def encode(self, text: str) -> None:
        return None

    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError

    def warm(self, segments: Sequence[Segment]) -> None:
        # One token is enough to get the transcript into the server's prefix cache.
        for _ in self.stream(segments, 1):
            pass

    def stream(self, segments: Sequence[Segment], max_tokens: int,
               interactive: bool = False) -> Iterator[Piece]:
        body = {"model": self.model, "prompt": "".join(t for t, _ in segments),
                "max_tokens": max_tokens, "stream": True, **self.params}
        req = urllib.request.Request(self.base + "/completions", data=json.dumps(body).encode(),
                                     headers=self.headers)
        deadline = time.monotonic() + self.INTERACTIVE_DEADLINE
        try:
            resp = urllib.request.urlopen(req, timeout=self.INTERACTIVE_READ if interactive else 120)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{e.code} from {self.base}: {e.read().decode(errors='replace')[:300]}") from e
        with resp:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    if interactive:
                        if time.monotonic() > deadline:
                            return
                        yield "", None  # keepalives and blank lines still hand back control
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                choices = json.loads(data).get("choices") or [{}]
                text = choices[0].get("text") or ""
                if text or interactive:
                    yield text, None
                if interactive and time.monotonic() > deadline:
                    return


def make_engine(model: str, url: str | None, temperature: float, top_p: float,
                api_key: str | None = None, bits: int = 4) -> Engine:
    if url:
        return RemoteEngine(url, model, temperature, top_p, api_key)
    try:
        import mlx_lm  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise SystemExit("[kenoma: the local engine needs mlx-lm (Apple silicon). Elsewhere, run "
                         "an OpenAI-compatible server (vLLM, llama-server, ...) and pass --url]")
    return MLXEngine(model, temperature, top_p, bits)
