"""
Transform for HC3 dataset.

Raw data format (jsonl):
    - question: str
    - human_answers: list[str] (human-written answers, typically one, but can be multiple)
    - chatgpt_answers: list[str] (ChatGPT-generated answers, typically one, but can be multiple)

Output: list[BinarySample]
"""

import os
from pathlib import Path

from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import BinarySample
from ..readers import read_jsonl
from ..constants import DATASET_DIR_OTHERS, HC3_EN_DIRECTORY

LABEL_MAPPING = {
    "human": 0,
    "chatgpt": 1,
}

def _process_answer(
        answer: str,
        answer_type: str,
) -> BinarySample | None:
    """
    Process a single answer into a BinarySample.

    Args:
        answer: The answer text.
        answer_type: "human" or "chatgpt".

    Returns:
        BinarySample instance if valid, None otherwise.
    """
    if not answer or len(answer.strip()) < 10:
        return None

    text = answer
    label = LABEL_MAPPING[answer_type]

    return BinarySample(
        text=text,
        label=label,
    )


def _process_file(file_path: Path) -> list[BinarySample]:
    """
    Process a single JSONL file and convert to BinarySample list.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of BinarySample instances.
    """
    samples = []

    try:
        records = read_jsonl(str(file_path))
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")
        return samples

    for record in records:
        human_answers = record.get("human_answers", [])
        chatgpt_answers = record.get("chatgpt_answers", [])

        # Process human answers
        for answer in human_answers:
            sample = _process_answer(answer, "human")
            if sample:
                samples.append(sample)

        # Process ChatGPT answers
        for answer in chatgpt_answers:
            sample = _process_answer(answer, "chatgpt")
            if sample:
                samples.append(sample)

    return samples


@DatasetRegistry.register("HC3")
class HC3Transform(DatasetTransform):
    """
    Transform for HC3 dataset (binary classification).

    Reads a jsonl with human answers and chatgpt answers,
    filters by text length, and produces BinarySample pairs.
    """

    output_schema = BinarySample

    def transform(self, raw_data: list[dict], **kwargs) -> list[BinarySample]:
        """
        Transform HC3 raw records into BinarySamples.

        Args:
            raw_data: List of dicts from jsonl. If empty, reads from default path.
            **kwargs:
                targetLLM (str): Name of the target LLM.
                path (str, optional): Override path to jsonl file.

        Returns:
            List of BinarySample instances.
        """
        all_samples = []

        targetLLM = kwargs.get("targetLLM")
        if targetLLM is None:
            raise ValueError("NarrativeQA transform requires 'targetLLM' parameter")

        # If raw_data is not provided, read from files
        if not raw_data:
            # directory = HC3_EN_DIRECTORY
            directory = kwargs.get("path") or os.path.join(
                DATASET_DIR_OTHERS, "HC3_EN"
            )
            directory = Path(directory)
            # Find all JSONL files
            jsonl_files = list(directory.glob("*.jsonl"))
            for file_path in jsonl_files:
                samples = _process_file(file_path)
                all_samples.extend(samples)

        else:
            for record in raw_data:
                human_answers = record.get("human_answers", [])
                chatgpt_answers = record.get("chatgpt_answers", [])

                for answer in human_answers:
                    sample = _process_answer(answer, "human")
                    if sample:
                        all_samples.append(sample)

                for answer in chatgpt_answers:
                    sample = _process_answer(answer, "chatgpt")
                    if sample:
                        all_samples.append(sample)

        return all_samples



