"""Dependency-free tests for the fixed GSM8K prompt and scorer."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from .evaluate import configure_rank_local_hf_modules_cache
from .protocol import (
    FEWSHOT_SAMPLES,
    NUM_SAMPLES,
    SCORER_NAME,
    SEED,
    build_prompt,
    exact_match,
    extract_olmes_answer,
    pass_at_1,
    truncate_at_stop,
)


class ProtocolTest(unittest.TestCase):
    def test_hf_modules_cache_is_local_rank_scoped(self) -> None:
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=True):
            cache = configure_rank_local_hf_modules_cache(Path("/tmp/gsm8k-output"))
        self.assertEqual(
            cache,
            str(
                Path("/tmp/gsm8k-output/cache/hf-modules/local-rank-03").resolve()
            ),
        )

    def test_prompt_has_exactly_eight_fixed_demonstrations(self) -> None:
        prompt = build_prompt("How many?")
        self.assertEqual(len(FEWSHOT_SAMPLES), 8)
        self.assertEqual(prompt.count("Question:"), 9)
        self.assertTrue(prompt.endswith("Question: How many?\nAnswer:"))
        self.assertEqual(NUM_SAMPLES, 1)
        self.assertEqual(SEED, 1234)

    def test_olmes_scorer_uses_last_number(self) -> None:
        self.assertEqual(extract_olmes_answer("The answer is 460 dollars."), "460")
        self.assertTrue(exact_match("4 + 12 + 6 = 22.", "#### 22"))

    def test_olmes_scorer_preserves_decimal_string_exactness(self) -> None:
        self.assertFalse(exact_match("The answer is 57.00.", "#### 57"))

    def test_pass_at_1_averages_repeated_samples(self) -> None:
        self.assertEqual(pass_at_1(("Answer: 5", "Answer: 4"), "#### 5"), 0.5)

    def test_stop_truncation_uses_earliest_stop(self) -> None:
        self.assertEqual(
            truncate_at_stop("The answer is 5.\\n\\nQuestion: next"),
            "The answer is 5.\\n\\n",
        )
        self.assertEqual(SCORER_NAME, "olmes_last_numeric_exact_match_v1")


if __name__ == "__main__":
    unittest.main()
