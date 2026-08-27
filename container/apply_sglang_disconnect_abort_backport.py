#!/usr/bin/env python3
"""Apply the SGLang disconnected-stream abort fix to the pinned runtime.

Rootcore intentionally pins SGLang at 5a7b26c63 for its validated Blackwell
stack. That revision drops TokenizerManager request state when a streaming
client disconnects before the delayed abort task runs. The scheduler then
keeps decoding an orphaned request until max_tokens.

This is a fail-closed backport of sgl-project/sglang#36418. It only accepts
the exact pinned source, applies every expected edit atomically, and supports
an idempotent --check mode for container builds.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import py_compile
import sys
import tempfile
from pathlib import Path


PINNED_SOURCE_SHA256 = (
    "08165ccaed9d5c3da34acf17d0b62b0e338c88eba083d585a756c80becab7df6"
)

REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "track scheduler-dispatched request ids",
        """        self._init_req_state(obj, request)\n        try:\n            if self.server_args.language_only:\n""",
        """        self._init_req_state(obj, request)\n        try:\n            dispatched_rids = set()\n            # The delayed disconnect abort must target the scheduler request IDs,\n            # including batch and parallel-sampling IDs generated below.\n            obj._dispatched_rids = dispatched_rids\n            if self.server_args.language_only:\n""",
    ),
    (
        "mark single request dispatched",
        """                    self._send_one_request(tokenized_obj)\n                    async for response in self._wait_one_response(obj, request):\n""",
        """                    self._send_one_request(tokenized_obj)\n                    dispatched_rids.add(obj.rid)\n                    async for response in self._wait_one_response(obj, request):\n""",
    ),
    (
        "pass dispatch tracker to batch handler",
        """                else:\n                    async for response in self._handle_batch_request(obj, request):\n                        yield response\n        except BaseException:\n""",
        """                else:\n                    async for response in self._handle_batch_request(\n                        obj, request, dispatched_rids\n                    ):\n                        yield response\n        except (asyncio.CancelledError, GeneratorExit):\n            # Retain scheduler-owned state until the forced abort is dispatched.\n            # Removing it here suppresses the delayed abort and creates a zombie.\n            self._discard_pending_req_states(obj, dispatched_rids)\n            raise\n        except BaseException:\n""",
    ),
    (
        "accept dispatch tracker in batch handler",
        """    async def _handle_batch_request(\n        self,\n        obj: Union[GenerateReqInput, EmbeddingReqInput],\n        request: Optional[fastapi.Request] = None,\n    ):\n""",
        """    async def _handle_batch_request(\n        self,\n        obj: Union[GenerateReqInput, EmbeddingReqInput],\n        request: Optional[fastapi.Request] = None,\n        dispatched_rids: Optional[set[str]] = None,\n    ):\n""",
    ),
    (
        "mark tokenized batch dispatched",
        """                self._send_batch_request(tokenized_objs)\n\n                # Set up generators for each request in the batch\n""",
        """                self._send_batch_request(tokenized_objs)\n                if dispatched_rids is not None:\n                    dispatched_rids.update(\n                        tokenized_obj.rid for tokenized_obj in tokenized_objs\n                    )\n\n                # Set up generators for each request in the batch\n""",
    ),
    (
        "mark sequential batch item dispatched",
        """                        self._send_one_request(tokenized_obj)\n                        generators.append(self._wait_one_response(tmp_obj, request))\n                        rids.append(tmp_obj.rid)\n""",
        """                        self._send_one_request(tokenized_obj)\n                        if dispatched_rids is not None:\n                            dispatched_rids.add(tmp_obj.rid)\n                        generators.append(self._wait_one_response(tmp_obj, request))\n                        rids.append(tmp_obj.rid)\n""",
    ),
    (
        "mark parallel prefix request dispatched",
        """                self._init_req_state(tmp_obj)\n                self._send_one_request(tokenized_obj)\n                await self._wait_one_response(tmp_obj, request).__anext__()\n""",
        """                self._init_req_state(tmp_obj)\n                self._send_one_request(tokenized_obj)\n                if dispatched_rids is not None:\n                    dispatched_rids.add(tokenized_obj.rid)\n                await self._wait_one_response(tmp_obj, request).__anext__()\n""",
    ),
    (
        "mark parallel sample dispatched",
        """                    self._send_one_request(tokenized_obj)\n                    generators.append(self._wait_one_response(tmp_obj, request))\n                    rids.append(tmp_obj.rid)\n""",
        """                    self._send_one_request(tokenized_obj)\n                    if dispatched_rids is not None:\n                        dispatched_rids.add(tokenized_obj.rid)\n                    generators.append(self._wait_one_response(tmp_obj, request))\n                    rids.append(tmp_obj.rid)\n""",
    ),
    (
        "force scheduler abort after local cancellation",
        """    def abort_request(self, rid: str = "", abort_all: bool = False):\n""",
        """    def abort_request(\n        self, rid: str = "", abort_all: bool = False, force: bool = False\n    ):\n""",
    ),
    (
        "bypass missing-state guard for forced abort",
        """        if (\n            not abort_all\n            and self.server_args.tokenizer_worker_num == 1\n            and rid not in self.rid_to_state\n        ):\n""",
        """        if (\n            not abort_all\n            and not force\n            and self.server_args.tokenizer_worker_num == 1\n            and rid not in self.rid_to_state\n        ):\n""",
    ),
    (
        "dispatch delayed abort for tracked ids",
        """        async def abort_request():\n            await asyncio.sleep(2)\n            if obj.is_single:\n                self.abort_request(obj.rid)\n            else:\n                for rid in obj.rid:\n                    self.abort_request(rid)\n""",
        """        async def abort_request():\n            await asyncio.sleep(2)\n            dispatched_rids = getattr(obj, "_dispatched_rids", None)\n            if dispatched_rids is not None:\n                # Cancellation can finish local cleanup before this task runs.\n                for rid in dispatched_rids:\n                    self.abort_request(rid, force=True)\n            elif obj.is_single:\n                self.abort_request(obj.rid)\n            else:\n                for rid in obj.rid:\n                    self.abort_request(rid)\n""",
    ),
    (
        "retain dispatched request state",
        """    def _discard_pending_req_states(self, obj):\n""",
        """    def _discard_pending_req_states(self, obj, dispatched_rids=None):\n""",
    ),
    (
        "discard only undispatched request state",
        """        for rid in rids:\n            self.rid_to_state.pop(rid, None)\n""",
        """        for rid in rids:\n            if dispatched_rids is None or rid not in dispatched_rids:\n                self.rid_to_state.pop(rid, None)\n""",
    ),
)


class BackportError(RuntimeError):
    """Raised when the pinned source is absent, drifted, or partially patched."""


def sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def patch_state(source: str) -> str:
    old_present = [old in source for _, old, _ in REPLACEMENTS]
    new_present = [new in source for _, _, new in REPLACEMENTS]
    if all(new_present) and not any(old_present):
        return "patched"
    if all(old_present) and not any(new_present):
        return "unpatched"
    return "drifted"


def transform_source(source: str, *, enforce_pin: bool = True) -> str:
    state = patch_state(source)
    if state == "patched":
        return source
    if state != "unpatched":
        missing = [
            label
            for (label, old, new) in REPLACEMENTS
            if (old in source) == (new in source)
        ]
        raise BackportError("partial patch or source drift: " + ", ".join(missing))
    if enforce_pin and sha256_text(source) != PINNED_SOURCE_SHA256:
        raise BackportError(
            "unrecognized tokenizer_manager.py; expected pinned SHA-256 "
            f"{PINNED_SOURCE_SHA256}, got {sha256_text(source)}"
        )

    result = source
    for label, old, new in REPLACEMENTS:
        if result.count(old) != 1:
            raise BackportError(f"{label}: expected exactly one source match")
        result = result.replace(old, new, 1)
    if patch_state(result) != "patched":
        raise BackportError("post-transform invariant failed")
    return result


def resolve_target(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    spec = importlib.util.find_spec("sglang.srt.managers.tokenizer_manager")
    if spec is None or spec.origin is None:
        raise BackportError("could not locate installed SGLang tokenizer_manager.py")
    return Path(spec.origin)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="path to tokenizer_manager.py")
    parser.add_argument(
        "--check", action="store_true", help="verify the backport without writing"
    )
    args = parser.parse_args()

    target = resolve_target(args.target)
    source = target.read_text()
    state = patch_state(source)
    if args.check:
        if state != "patched":
            raise BackportError(f"backport check failed: source is {state}")
        py_compile.compile(str(target), doraise=True)
        print(f"sglang-disconnect-abort-backport: verified {target}")
        return 0

    result = transform_source(source)
    if result != source:
        with tempfile.NamedTemporaryFile(
            "w", dir=target.parent, prefix=target.name + ".", delete=False
        ) as handle:
            handle.write(result)
            temp_path = Path(handle.name)
        temp_path.chmod(target.stat().st_mode)
        temp_path.replace(target)
    py_compile.compile(str(target), doraise=True)
    print(f"sglang-disconnect-abort-backport: applied {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BackportError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
