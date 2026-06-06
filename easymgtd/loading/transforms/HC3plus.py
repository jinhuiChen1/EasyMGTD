"""
Transform for HC3plus dataset.

Raw data format (jsonl):
    - text: str
    - label: int

Output: list[BinarySample]

Default path for dataset: DATASET_DIR_OTHERS/HC3plus/en  or  DATASET_DIR_OTHERS/HC3plus/zh
"""

import os
from pathlib import Path

from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import BinarySample
from ..readers import read_jsonl
from ..constants import DATASET_DIR_OTHERS


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
        text = record.get("text")
        label = record.get("label")

        samples.append(BinarySample(text, label))

    return samples

@DatasetRegistry.register("HC3plus")
class HC3plusTransform(DatasetTransform):
    """
    Transform for HC3plus dataset (binary classification).

    """

    output_schema = BinarySample

    def transform(self, raw_data: list[dict], **kwargs) -> list[BinarySample]:
        """
        Transform HC3-PLUS raw records into BinarySamples.

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
            directory = kwargs.get("path") or os.path.join(
                DATASET_DIR_OTHERS, "HC3plus/en"
            )
            directory = Path(directory)

            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
            # Find all JSONL files
            jsonl_files = list(directory.glob("*.jsonl"))
            for file_path in jsonl_files:
                samples = _process_file(file_path)
                all_samples.extend(samples)

        else:
            for record in raw_data:
                text = record.get("text")
                label = record.get("label")

                all_samples.append(BinarySample(text, label))

        return all_samples