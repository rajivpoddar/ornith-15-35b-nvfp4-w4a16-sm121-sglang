from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PATCHER_PATH = (
    Path(__file__).resolve().parents[1] / "apply_sglang_disconnect_abort_backport.py"
)
SPEC = importlib.util.spec_from_file_location("sglang_backport", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


class TestSglangDisconnectAbortBackport(unittest.TestCase):
    def setUp(self) -> None:
        self.unpatched = "\n# boundary\n".join(
            old for _, old, _ in PATCHER.REPLACEMENTS
        )

    def test_exact_unpatched_contract_becomes_fully_patched(self) -> None:
        patched = PATCHER.transform_source(self.unpatched, enforce_pin=False)

        self.assertEqual(PATCHER.patch_state(patched), "patched")
        for _, old, new in PATCHER.REPLACEMENTS:
            self.assertNotIn(old, patched)
            self.assertIn(new, patched)

    def test_abort_ids_are_recorded_only_on_cancellation(self) -> None:
        patched = PATCHER.transform_source(self.unpatched, enforce_pin=False)

        self.assertEqual(
            patched.count("obj._dispatched_rids = dispatched_rids.copy()"), 1
        )
        self.assertNotIn("obj._dispatched_rids = dispatched_rids\n", patched)

    def test_partial_application_fails_closed(self) -> None:
        _, first_old, first_new = PATCHER.REPLACEMENTS[0]
        partial = self.unpatched.replace(first_old, first_new, 1)

        with self.assertRaisesRegex(PATCHER.BackportError, "partial patch"):
            PATCHER.transform_source(partial, enforce_pin=False)

    def test_unrecognized_source_sha_fails_closed(self) -> None:
        with self.assertRaisesRegex(PATCHER.BackportError, "unrecognized"):
            PATCHER.transform_source(self.unpatched, enforce_pin=True)

    def test_validation_does_not_write_bytecode(self) -> None:
        source = "value = 1\n"
        target = Path("/root-owned/site-packages/module.py")

        PATCHER.validate_python(source, target)


if __name__ == "__main__":
    unittest.main()
