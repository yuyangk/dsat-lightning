import math
import pickle
from datetime import timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from loguru import logger

from .image_processor import ImageProcessor

TCSA_H5_FILE_NAME = "TCSA.h5"

REGION_TO_CODE = {
    "WP": 0,
    "EP": 1,
    "AL": 2,
    "SH": 3,
    "IO": 4,
    "CP": 5,
}


class TCSAPreprocessor:
    def __init__(self, data_folder: str, year_list: list, coordinate: str = "cart"):
        self.data_folder = data_folder
        self.year_list = year_list
        self.coordinate = coordinate

    def preprocess(self) -> None:
        if not self._all_pickles_exist():
            logger.info("Not all pickles exist!")
            logger.info("Generating all pickles...")
            self._generate_all_pickles()
        else:
            logger.info("All pickles exist!")

    def _all_pickles_exist(self) -> bool:
        for year in self.year_list:
            pickle_path = Path(
                f"{self.data_folder}/TCSA.{year}.{self.coordinate}.pickle"
            )
            if not pickle_path.is_file():
                return False
        return True

    def _generate_all_pickles(self) -> None:
        tcsa_h5_path = Path(f"{self.data_folder}/{TCSA_H5_FILE_NAME}")
        if not tcsa_h5_path.is_file():
            logger.info(f"{tcsa_h5_path} not found! try to download it first!")
            raise FileNotFoundError(
                f"{tcsa_h5_path} not found! try to download it first!"
            )

        raw_image, raw_info, raw_profiles = self._load_raw_data_from_h5()
        processed_image = self._process_image(raw_image, raw_info)
        processed_feature, processed_label = self._process_feature(raw_info)

        self._save_pickle(
            processed_image, processed_feature, processed_label, raw_profiles
        )

    def _load_raw_data_from_h5(self) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        h5_path = Path(f"{self.data_folder}/{TCSA_H5_FILE_NAME}")
        with h5py.File(h5_path, "r") as hf:
            images = hf["images"][:]
            info = pd.read_hdf(h5_path, key="info", mode="r")
            profiles = hf["structure_profiles"][:]
        return images, info, profiles

    def _process_image(self, images: np.ndarray, info_df: pd.DataFrame) -> np.ndarray:
        images = self._remove_outlier_and_nan(images)
        images = self._flip_SH_images(info_df, images)

        if self.coordinate == "polar":
            images = ImageProcessor(images).cart2polar()

        return images

    def _process_feature(
        self, info_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        info_df["year"] = info_df.ID.map(lambda x: int(x[:4]))
        info_df["region_code"] = info_df["region"].map(REGION_TO_CODE).astype(int)
        info_df["lon"] = (
            info_df.lon + 180
        ) % 360 - 180  # calibrate longitude, ex: 190 -> -170
        info_df["GMT_time"] = pd.to_datetime(info_df.time, format="%Y%m%d%H")
        info_df["local_time"] = info_df.GMT_time + info_df.apply(
            lambda x: timedelta(hours=x.lon / 15), axis=1
        )
        # --- year_day ---
        SH_idx = info_df.index[info_df.region == "SH"]
        info_df["yday"] = info_df.local_time.apply(
            lambda x: float(x.timetuple().tm_yday)
        )
        info_df.loc[SH_idx, "yday"] += 365 / 2  # TC from SH
        info_df["yday_transform"] = info_df.yday.apply(lambda x: x / 365 * 2 * math.pi)
        info_df["yday_sin"] = info_df.yday_transform.apply(lambda x: math.sin(x))
        info_df["yday_cos"] = info_df.yday_transform.apply(lambda x: math.cos(x))

        # --- hour ---
        info_df["hour_transform"] = info_df.apply(
            lambda x: x.local_time.hour / 24 * 2 * math.pi, axis=1
        )
        info_df["hour_sin"] = info_df.hour_transform.apply(lambda x: math.sin(x))
        info_df["hour_cos"] = info_df.hour_transform.apply(lambda x: math.cos(x))
        info_df["minutes_to_noon"] = info_df.local_time.apply(self._get_minutes_to_noon)

        info_df["is_good_quality_VIS"] = False  # should be applied if VIS is needed

        # split into 2 dataframe
        label_columns = [
            "year",
            "region",
            "ID",
            "local_time",
            "Vmax",
            "R34",
            "MSLP",
            "valid_profile",
        ]
        feature_columns = [
            "lon",
            "lat",
            "region_code",
            "yday_cos",
            "yday_sin",
            "hour_cos",
            "hour_sin",
            "minutes_to_noon",
            "is_good_quality_VIS",
        ]

        label_df = info_df[label_columns]
        feature_df = info_df[feature_columns]
        return feature_df, label_df

    def _save_pickle(
        self,
        images: np.ndarray,
        feature: pd.DataFrame,
        label: pd.DataFrame,
        profiles: np.ndarray,
    ) -> None:
        for y, year_df in label.groupby("year"):
            target_index = year_df.index
            year_dataset = {
                "label": year_df.drop(["year"], axis=1).reset_index(drop=True),
                "feature": feature.iloc[target_index].reset_index(drop=True),
                "image": images[target_index],
                "profile": profiles[target_index],
            }
            print(f"saving year {y} data pickle!")
            save_path = Path(f"{self.data_folder}/TCSA.{y}.{self.coordinate}.pickle")
            with open(save_path, "wb") as save_file:
                pickle.dump(year_dataset, save_file, protocol=5)

    @staticmethod
    def _get_minutes_to_noon(local_time):
        minutes_in_day = 60 * local_time.hour + local_time.minute
        noon = 60 * 12
        return abs(noon - minutes_in_day)

    @staticmethod
    def _remove_outlier_and_nan(numpy_array, upper_bound=1000):
        numpy_array = np.nan_to_num(numpy_array, copy=False)
        numpy_array[numpy_array > upper_bound] = 0
        VIS = numpy_array[:, :, :, 2]
        VIS[VIS > 1] = 1  # VIS channel ranged from 0 to 1
        return numpy_array

    @staticmethod
    def _flip_SH_images(info_df, image_matrix):
        SH_idx = info_df.index[info_df.region == "SH"]
        image_matrix[SH_idx] = np.flip(image_matrix[SH_idx], 1)
        return image_matrix
