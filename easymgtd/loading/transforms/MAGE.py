"""
Transform for MAGE dataset.

Raw data format (CSV):
    - text: str
    - label: int

Output: list[BinarySample]
"""

import os
from pathlib import Path

from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import BinarySample
from ..readers import read_csv
from ..constants import DATASET_DIR_OTHERS, MAGE_DIRECTORY


def _process_file(file_path: Path) -> list[BinarySample]:
    """
    Process a single CSV file and convert to BinarySample list.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of BinarySample instances.
    """
    samples = []

    try:
        records = read_csv(str(file_path))
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")
        return samples

    for record in records:
        text = str(record.get("text", ""))
        label = record.get("label", 0)

        samples.append(BinarySample(text, label))

    return samples

@DatasetRegistry.register("MAGE")
class MAGETransform(DatasetTransform):
    """
    Transform for MAGE dataset (binary classification).

    """

    output_schema = BinarySample

    def transform(self, raw_data: list[dict], **kwargs) -> list[BinarySample]:
        """
        Transform MAGE raw records into BinarySamples.

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
            # directory = MAGE_DIRECTORY
            directory = kwargs.get("path") or os.path.join(
                DATASET_DIR_OTHERS, "MAGE"
            )
            directory = Path(directory)
            # Find all CSV files
            csv_files = list(directory.glob("*.csv"))
            for file_path in csv_files:
                samples = _process_file(file_path)
                all_samples.extend(samples)

        else:
            for record in raw_data:
                text = record.get("text")
                label = record.get("label")

                all_samples.append(BinarySample(text, label))

        return all_samples
