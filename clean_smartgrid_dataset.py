from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "DataSet" / "RAW_smartgrid_5yr_hourly_natural.csv"
OUTPUT_FILE = ROOT / "DataSet" / "CLEAN_smartgrid_5yr_hourly_natural.csv"

# Keep the core variables needed for a forecasting / load-analysis workflow.
# Some columns are redundant calendar-derived features and can be regenerated from timestamp.
CORE_COLUMNS = [
    "timestamp",
    "temperature",
    "is_rain_day",
    "holiday_type_national",
    "holiday_type_religious",
    "is_lockdown",
    "load_kw",
]


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Input file must contain a 'timestamp' column.")
    return df


def parse_and_validate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    invalid_ts = df["timestamp"].isna().sum()
    if invalid_ts:
        print(f"Warning: {invalid_ts} invalid timestamps found and will be dropped.")
        df = df.dropna(subset=["timestamp"]).copy()

    return df


def sort_and_check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)

    duplicate_count = int(df["timestamp"].duplicated().sum())
    if duplicate_count > 0:
        print(f"Warning: {duplicate_count} duplicate timestamps found. Keeping the last occurrence.")
        df = df.drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)

    return df


def check_missing_time_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    existing_index = pd.DatetimeIndex(df["timestamp"])

    missing = full_index.difference(existing_index)
    if len(missing) > 0:
        print(f"Warning: {len(missing)} missing timestamps found. Reindexing to complete hourly series.")

        df = df.set_index("timestamp").reindex(full_index).reset_index()
        df = df.rename(columns={"index": "timestamp"})
    else:
        df = df.copy()

    return df


def handle_missing_observations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    missing_summary = df.isna().sum()
    missing_columns = missing_summary[missing_summary > 0]
    if missing_columns.empty:
        print("No missing observations found in the dataset.")
        return df

    print("Missing observations detected:")
    print(missing_columns)

    for column in df.columns:
        if column == "timestamp":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].interpolate(method="time", limit_direction="both")
        else:
            df[column] = df[column].ffill().bfill()

    return df


def select_variables(df: pd.DataFrame) -> pd.DataFrame:
    available = set(df.columns)
    selected = [col for col in CORE_COLUMNS if col in available]
    missing_requested = [col for col in CORE_COLUMNS if col not in available]
    if missing_requested:
        raise ValueError(f"Requested output columns are missing from input: {missing_requested}")

    return df[selected].copy()


def main() -> None:
    print(f"Loading raw file: {RAW_FILE}")
    raw_df = load_raw_data(RAW_FILE)

    print("Step 1: parse timestamp")
    df = parse_and_validate_timestamps(raw_df)

    print("Step 2: sort chronologically")
    df = sort_and_check_duplicates(df)

    print("Step 3: check duplicate timestamps")
    duplicate_count = int(df["timestamp"].duplicated().sum())
    if duplicate_count > 0:
        print(f"Duplicate timestamps still present after sorting: {duplicate_count}")
    else:
        print("No duplicate timestamps remain.")

    print("Step 4: check missing timestamps")
    df = check_missing_time_index(df)

    print("Step 5: handle missing observations")
    df = handle_missing_observations(df)

    print("Step 6: select variables")
    df = select_variables(df)

    print(f"Output shape: {df.shape}")
    print(df.head().to_string(index=False))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Step 7: cleaned dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
