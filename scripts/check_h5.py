import h5py
import pandas as pd


def check_chunk(file_path):
    with h5py.File(file_path, "r") as f:
        dataset = f["images"]

        # 檢查結果
        if dataset.chunks is None:
            print("❌ 這是一個 Contiguous (連續) 資料集，沒有做 Chunking！")
            print("這會導致隨機讀取非常緩慢。")
        else:
            print(f"✅ 有設定 Chunking！塊的大小為: {dataset.chunks}")
            print(f"資料形狀 (Shape) 為: {dataset.shape}")


def check_h5_size(file_path):
    with h5py.File(file_path, "r") as f:
        print(f"File: {file_path}\n")

        # Images size
        images = f["images"]
        print(f"images:             shape={images.shape}, dtype={images.dtype}")

        # Structure profiles size
        structure_profiles = f["structure_profiles"]
        print(
            f"structure_profiles: shape={structure_profiles.shape}, dtype={structure_profiles.dtype}"
        )

    # Info columns (pandas DataFrame; load only schema to get column names)
    info_df = pd.read_hdf(file_path, key="info")
    # print the col name and info size
    print(info_df.columns)


def check_h5_keys(file_path):
    with h5py.File(file_path, "r") as f:
        print(f.keys())


def check_h5_size_memory(file_path):
    with h5py.File(file_path, "r") as f:
        ds = f["images"]
        # 預估 GB
        print(f"預估全讀所需記憶體: {ds.nbytes / (1024**3):.2f} GB")


def check_region_code(file_path):
    info_df = pd.read_hdf(file_path, key="info")
    print(info_df["region"].unique())
    # count the region code
    print(info_df["region"].value_counts())

    # use pd categorical to count the region code
    info_df["region_code"] = pd.Categorical(info_df.region).codes
    # print the region code and region pair
    print(info_df[["region_code", "region"]].value_counts())


def main():
    file_path = "data/TCSA_2004_2018.h5"
    check_region_code(file_path)


if __name__ == "__main__":
    main()
