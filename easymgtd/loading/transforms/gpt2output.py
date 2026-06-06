"""
Transform for GPT-2 Output Dataset.

This dataset contains JSONL files with human-written (webtext) and GPT-2 generated text.
Dataset URL: https://github.com/openai/gpt-2-output-dataset

File naming pattern:
    - webtext.{split}.jsonl: human-written text
    - {model_name}.{split}.jsonl: GPT-2 generated text (e.g., small-117M, medium-345M,
      large-762M, xl-1542M)
    - {model_name}-k40.{split}.jsonl: GPT-2 generated text with Top-K 40 truncation

Each line in the JSONL file is a JSON object with a 'text' field.

Default path for dataset: DATASET_DIR_OTHERS/GPT2Output
"""
import os
from pathlib import Path

from ..constants import DATASET_DIR_OTHERS
from ..registry import DatasetRegistry, DatasetTransform
from ..schemas import MultiClassSample
from ..readers import read_jsonl

_label_mapping = {
    "human": 0,  # WebText
    "small-117M": 1,
    "small-117M-k40": 2,
    "medium-345M": 3,
    "medium-345M-k40": 4,
    "large-762M": 5,
    "large-762M-k40": 6,
    "xl-1542M": 7,
    "xl-1542M-k40": 8,
}

# 用于 category 字段的显示名称
_category_mapping = {
    "human": "WebText",
    "small-117M": "GPT-2 Small (117M)",
    "small-117M-k40": "GPT-2 Small (117M, Top-K 40)",
    "medium-345M": "GPT-2 Medium (345M)",
    "medium-345M-k40": "GPT-2 Medium (345M, Top-K 40)",
    "large-762M": "GPT-2 Large (762M)",
    "large-762M-k40": "GPT-2 Large (762M, Top-K 40)",
    "xl-1542M": "GPT-2 XL (1542M)",
    "xl-1542M-k40": "GPT-2 XL (1542M, Top-K 40)",
}

splits = ["train", "test", "valid"]

def _extract_model_info(file_path: Path) -> tuple:
    """
    Extract model name and split from the filename.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        Tuple of (model_name, split) where model_name is the key used in
        MODEL_TO_LABEL mapping, e.g., 'small-117M', 'large-762M-k40'.
    """
    # Remove .jsonl extension
    basename = file_path.stem
    # Split by '.' to separate model name and split
    parts = basename.rsplit(".", 1)

    if len(parts) == 2:
        model_name = parts[0]
        split = parts[1]
    else:
        model_name = basename
        split = "unknown"

    # Convert 'webtext' to 'human' for internal mapping
    if model_name == "webtext":
        model_name = "human"

    return model_name, split

def _scan_dataset_files(dir_path: str) -> list[dict[str, any]]:
    """
    Scan the dataset directory for all JSONL files.

    Returns:
        List of dictionaries with keys: 'path', 'model_name', 'split', 'label', 'category'
    """
    file_info_list = []
    directory = dir_path or os.path.join(DATASET_DIR_OTHERS, "GPT2Output")
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for file_path in directory.glob("*.jsonl"):
        model_name, split = _extract_model_info(file_path)

        file_info_list.append({
            "path": file_path,
            "model_name": model_name,
            "split": split,
            "label": _label_mapping[model_name],
            "category": _category_mapping[model_name],
        })

    return file_info_list

@DatasetRegistry.register("GPT2Output")
class GPT2OutputTransform(DatasetTransform):
    """
    Transform for GPT-2 Output Dataset (multi-class classification).

    Reads all JSONL files in the dataset directory, determines model/source from
    the filename, and produces MultiClassSample instances with appropriate labels.

    """

    output_schema = MultiClassSample

    def transform(self, raw_data: list[dict], **kwargs) -> list[MultiClassSample]:
        """
        Transform GPT-2 Output Dataset into MultiClassSamples.

        Args:
            raw_data: List of dicts from JSONL. If not provided, reads from dataset_path.
            **kwargs: Additional arguments (path can be passed to override).

        Returns:
            List of MultiClassSample instances.
        """
        samples = []

        targetLLM = kwargs.get("targetLLM")
        if targetLLM is None:
            raise ValueError("GPT2Output transform requires 'targetLLM' parameter")

        # If raw_data is not provided, read from files
        if not raw_data:

            file_info_list = _scan_dataset_files(kwargs.get("path"))

            for file_info in file_info_list:
                file_path = file_info["path"]
                label = file_info["label"]
                category = file_info["category"]

                # Read the JSONL file
                try:
                    records = read_jsonl(str(file_path))
                except Exception as e:
                    print(f"Warning: Failed to read {file_path}: {e}")
                    continue

                for record in records:
                    text = record.get("text", "")
                    # Apply minimum length filter
                    if not text or len(text.strip()) < 10:
                        continue

                    samples.append(
                        MultiClassSample(
                            text=text,
                            label=label,
                            category=category
                        )
                    )

        else:
            for row in raw_data:
                text = row.get("text", "")
                if not text or len(text.strip()) < 10:
                    continue
                label = row.get("label")
                category = row.get("category")

                samples.append(
                    MultiClassSample(
                        text=text,
                        label=label,
                        category=category
                    )
                )

        return samples