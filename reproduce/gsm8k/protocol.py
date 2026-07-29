"""Fixed OLMo/OLMES GSM8K prompt and scoring protocol."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

TASK_NAME = "olmo_eval_paper_gsm8k_main"
DATASET_ID = "openai/gsm8k"
DATASET_NAME = "main"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
EXPECTED_TEST_SIZE = 1319
EXPECTED_PROMPT_CORPUS_SHA256 = (
    "03423a681bb3c3571df5c22d7cb5338adaa25fa14c17a149e96fee3f11d88d5e"
)
EXPECTED_EVALUATION_CORPUS_SHA256 = (
    "376b4fbe48a07ac1d75cc4db6ea1358740b0cb65f77c9793659e6d843bf23871"
)
NUM_FEWSHOT = 8
NUM_SAMPLES = 1
SEED = 1234
MAX_NEW_TOKENS = 512
MAX_SEQUENCE_LENGTH = 4096
TEMPERATURE = 0.6
TOP_P = 0.6
STOP_STRINGS = ("Question:", "</s>", "<|im_end|>")
SCORER_NAME = "olmes_last_numeric_exact_match_v1"

FEWSHOT_SAMPLES = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the "
        "grove today. After they are done, there will be 21 trees. How many trees "
        "did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more "
        "were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many "
        "cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
        "pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they "
        "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and "
        "dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, "
        "then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from monday to thursday. How many computers are now "
        "in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers "
        "were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer "
        "is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the end of "
        "wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had "
        "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The "
        "answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does "
        "she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 "
        "dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)

_NUMBER_RE = re.compile(r"[-+]?\d*\.\d+|\d+")


def build_prompt(question: str) -> str:
    """Render the exact fixed first-N eight-shot GSM8K prompt."""

    demonstrations = "\n\n".join(
        f"Question: {example_question}\nAnswer: {example_answer}"
        for example_question, example_answer in FEWSHOT_SAMPLES
    )
    return f"{demonstrations}\n\nQuestion: {question}\nAnswer:"


def prompt_corpus_sha256(prompts: Sequence[str]) -> str:
    """Hash ordered prompts with length framing to detect protocol drift."""

    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def evaluation_corpus_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    """Hash ordered questions and gold answers to detect dataset drift."""

    digest = hashlib.sha256()
    for row in rows:
        for field in ("question", "answer"):
            encoded = str(row[field]).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def extract_olmes_answer(text: str) -> str:
    """Apply the OLMES GSM8K last-numeric-span extraction contract."""

    without_digit_grouping = re.sub(r"(\d),(\d)", r"\1\2", str(text))
    matches = _NUMBER_RE.findall(without_digit_grouping)
    return matches[-1] if matches else without_digit_grouping


def exact_match(prediction: str, gold: str) -> bool:
    """Score one sampled output against its GSM8K target."""

    return extract_olmes_answer(prediction) == extract_olmes_answer(gold)


def pass_at_1(predictions: Sequence[str], gold: str) -> float:
    """Average single-sample correctness over repeated samples."""

    if not predictions:
        return 0.0
    correct = sum(exact_match(prediction, gold) for prediction in predictions)
    return correct / len(predictions)


def truncate_at_stop(text: str) -> str:
    """Remove any decoded suffix beginning with the earliest configured stop."""

    stop_positions = [
        position
        for stop in STOP_STRINGS
        if (position := text.find(stop)) >= 0
    ]
    return text if not stop_positions else text[: min(stop_positions)]
