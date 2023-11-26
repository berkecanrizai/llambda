from pathlib import Path

from .load_utils import ICLDataset, split_icl_dataset


class GuideLoader:
    def __init__(self, examples: dict, train_test_split=None):
        self.dataset = ICLDataset(examples)

        if train_test_split is not None:
            self.dataset = split_icl_dataset(
                self.dataset, test_size=train_test_split, seed=0
            )

    @classmethod
    def from_dict(cls, guide_dict):
        return cls(guide_dict)

    @classmethod
    def from_dataset(cls, dataset_path: str | Path):
        dataset = ICLDataset(dataset_path)
        return cls(dataset)
