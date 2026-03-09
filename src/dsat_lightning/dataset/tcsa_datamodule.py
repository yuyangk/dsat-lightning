"""Lightning DataModule that loads TCSA data from yearly pickle files."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import lightning as L
import torch
from typing import Callable, Optional

from .tcsa_dataset import TCSADataset
from .tcsa_preprocessor import TCSAPreprocessor
from loguru import logger
from dsat_lightning.dataset.image_processor import ImageProcessor


def remove_bad_quality_VIS_data(
    tcsa_dataset: dict[str, pd.DataFrame | np.ndarray],
) -> dict[str, pd.DataFrame | np.ndarray]:
    good_VIS_index = tcsa_dataset["feature"].index[
        tcsa_dataset["feature"]["is_good_quality_VIS"]
    ]
    tcsa_dataset["label"] = (
        tcsa_dataset["label"].loc[good_VIS_index].reset_index(drop=True)
    )
    tcsa_dataset["feature"] = (
        tcsa_dataset["feature"].loc[good_VIS_index].reset_index(drop=True)
    )
    tcsa_dataset["image"] = tcsa_dataset["image"][good_VIS_index]
    tcsa_dataset["profile"] = tcsa_dataset["profile"][good_VIS_index]
    return tcsa_dataset


def remove_invalid_profile_data(
    tcsa_dataset: dict[str, pd.DataFrame | np.ndarray],
) -> dict[str, pd.DataFrame | np.ndarray]:
    valid_profile_index = tcsa_dataset["label"].index[
        tcsa_dataset["label"]["valid_profile"] > 0
    ]
    tcsa_dataset["label"] = (
        tcsa_dataset["label"].loc[valid_profile_index].reset_index(drop=True)
    )
    tcsa_dataset["feature"] = (
        tcsa_dataset["feature"].loc[valid_profile_index].reset_index(drop=True)
    )
    tcsa_dataset["image"] = tcsa_dataset["image"][valid_profile_index]
    tcsa_dataset["profile"] = tcsa_dataset["profile"][valid_profile_index]
    return tcsa_dataset


def phase_rules_to_years(phase_rules: dict | None) -> dict[str, list[int]]:
    """Interpret phase-rule dict into concrete year lists per split."""
    DEFAULT_PHASE_RULES = {
        "train": list(range(2004, 2015)),
        "valid": list(range(2015, 2017)),
        "test": list(range(2017, 2019)),
    }
    if not phase_rules:
        return DEFAULT_PHASE_RULES

    years_dict: dict[str, list[int]] = {}
    for phase, rules in phase_rules.items():
        years: set[int] = set()
        if "range" in rules:
            start, end = rules["range"]
            years |= set(range(start, end))
        if "exclude" in rules:
            years -= set(rules["exclude"])
        if "add" in rules:
            years |= set(rules["add"])
        years_dict[phase] = sorted(years)
    return years_dict


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class TCSADataModule(L.LightningDataModule):
    """Load TCSA data from yearly pickle files into train / valid / test splits.

    Parameters
    ----------
    data_folder:
        Directory containing the pickle (or raw h5) files.
    batch_size:
        Mini-batch size.
    num_workers:
        DataLoader worker processes.
    coordinate:
        ``'polar'`` (180 x 103) or ``'cart'`` (128 x 128).
    valid_profile_only:
        Keep only samples whose wind-speed profile is valid.
    good_vis_only:
        Keep only samples with good-quality VIS channel.
    phase_rules:
        Dict defining how years map to train / valid / test.
    is_random_rotation_when_training:
        Apply random rotation to the image when training.
    """

    def __init__(
        self,
        data_folder: str,
        batch_size: int = 20,
        num_workers: int = 4,
        coordinate: str = "polar",
        valid_profile_only: bool = True,
        good_vis_only: bool = False,
        phase_rules: dict[str, Any] | None = None,
        is_random_rotation_when_training: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._datasets: dict[str, TCSADataset] = {}

    def prepare_data(self) -> None:
        years_dict = phase_rules_to_years(self.hparams.phase_rules)
        for phase, year_list in years_dict.items():
            preprocessor = TCSAPreprocessor(
                self.hparams.data_folder, year_list, self.hparams.coordinate
            )
            preprocessor.preprocess()

    def setup(self, stage: str | None = None) -> None:
        """Prepare datasets for the current run stage.

        stage : optional
            Lightning run stage: ``"fit"`` (train+valid), ``"validate"`` (valid),
            ``"test"`` (test), or ``"predict"`` (test). If None, all phases are loaded.
            Only the phases needed for this stage are loaded (e.g. only test years when
            stage is ``"test"``).
        """
        data_folder = self.hparams.data_folder
        years_dict = phase_rules_to_years(self.hparams.phase_rules)

        # Which phases to load for this stage (None = load all)
        phases_to_load: list[str] | None
        if stage == "fit":
            phases_to_load = ["train", "valid"]
        elif stage == "validate":
            phases_to_load = ["valid"]
        elif stage in ("test", "predict"):
            phases_to_load = ["test"]
        else:
            phases_to_load = list(years_dict.keys())  # load all

        for phase, year_list in years_dict.items():
            if phase not in phases_to_load:
                continue
            if phase in self._datasets:
                continue  # already loaded (e.g. from a previous setup call)

            year_datasets = []
            for year in year_list:
                pickle_path = Path(
                    f"{data_folder}/TCSA.{year}.{self.hparams.coordinate}.pickle"
                )
                with open(pickle_path, "rb") as load_file:
                    year_dataset = pickle.load(load_file)
                year_datasets.append(year_dataset)

            print("concat separated data")
            returned_dataset = {
                "label": pd.concat(
                    [year_dataset["label"] for year_dataset in year_datasets],
                    ignore_index=True,
                ),
                "feature": pd.concat(
                    [year_dataset["feature"] for year_dataset in year_datasets],
                    ignore_index=True,
                ),
                "image": np.concatenate(
                    [year_dataset["image"] for year_dataset in year_datasets], axis=0
                ),
                "profile": np.concatenate(
                    [year_dataset["profile"] for year_dataset in year_datasets], axis=0
                ),
            }

            if self.hparams.good_vis_only:
                logger.info("Removing bad quality VIS data...")
                returned_dataset = remove_bad_quality_VIS_data(returned_dataset)
            if self.hparams.valid_profile_only:
                logger.info("Removing invalid profile data...")
                returned_dataset = remove_invalid_profile_data(returned_dataset)

            is_random_rotation = (
                self.hparams.is_random_rotation_when_training and phase == "train"
            )

            self._datasets[phase] = TCSADataset(
                images=returned_dataset["image"],
                features=returned_dataset["feature"].to_numpy(dtype="float32"),
                profiles=returned_dataset["profile"].astype(np.float32),
                vmaxs=returned_dataset["label"][["Vmax"]].to_numpy(dtype="float32"),
                r34s=returned_dataset["label"][["R34"]].to_numpy(dtype="float32"),
                is_random_rotation=is_random_rotation,
            )
            print(f"[{phase}] loaded {len(self._datasets[phase])} samples")

    def train_dataloader(self) -> DataLoader | None:
        if "train" not in self._datasets:
            return None
        return DataLoader(
            self._datasets["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | None:
        if "valid" not in self._datasets:
            return None
        return DataLoader(
            self._datasets["valid"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader | None:
        if "test" not in self._datasets:
            return None
        return DataLoader(
            self._datasets["test"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
