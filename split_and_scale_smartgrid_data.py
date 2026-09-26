"""Split the smart-grid data by time and scale its two numeric columns.

The data is never shuffled:
    - training:   2020-01-01 through 2022-12-31
    - validation: 2023-01-01 through 2023-12-31
    - test:       2024-01-01 onward

Important: the scaler learns its averages and standard deviations from the
training data only. This prevents information from validation or test data
from leaking into training.
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Files and column names used by this script.
PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_FOLDER / "DataSet" / "CLEAN_smartgrid_5yr_hourly_natural.csv"
OUTPUT_FOLDER = PROJECT_FOLDER / "DataSet" / "splits"

TIME_COLUMN = "timestamp"
NUMERIC_COLUMNS = ["temperature", "load_kw"]


def load_data() -> pd.DataFrame:
    """Read the cleaned CSV file and put rows in chronological order."""
    data = pd.read_csv(INPUT_FILE, parse_dates=[TIME_COLUMN])
    return data.sort_values(TIME_COLUMN).reset_index(drop=True)


def split_by_time(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return training, validation, and test data in chronological order."""
    # The end date is excluded. For example, training ends immediately before
    # 2023-01-01, so it contains every hour from 2020, 2021, and 2022.
    training_data = data[
        (data[TIME_COLUMN] >= "2020-01-01") & (data[TIME_COLUMN] < "2023-01-01")
    ].copy()
    validation_data = data[
        (data[TIME_COLUMN] >= "2023-01-01") & (data[TIME_COLUMN] < "2024-01-01")
    ].copy()
    test_data = data[data[TIME_COLUMN] >= "2024-01-01"].copy()

    if training_data.empty or validation_data.empty or test_data.empty:
        raise ValueError("A split is empty. Check that the input data covers 2020 onward.")

    return training_data, validation_data, test_data


def scale_numeric_columns(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale temperature and load_kw using statistics learned from training only."""
    scaler = StandardScaler()

    # Never call fit on validation_data or test_data. They must not influence
    # the values used to train a model.
    scaler.fit(training_data[NUMERIC_COLUMNS])

    # Copy each split before replacing its two numeric columns. All other
    # columns, including the 0/1 indicator columns, are left exactly as-is.
    scaled_training = training_data.copy()
    scaled_validation = validation_data.copy()
    scaled_test = test_data.copy()

    scaled_training[NUMERIC_COLUMNS] = scaler.transform(training_data[NUMERIC_COLUMNS])
    scaled_validation[NUMERIC_COLUMNS] = scaler.transform(validation_data[NUMERIC_COLUMNS])
    scaled_test[NUMERIC_COLUMNS] = scaler.transform(test_data[NUMERIC_COLUMNS])

    return scaled_training, scaled_validation, scaled_test, scaler


def print_split_details(split_name: str, data: pd.DataFrame) -> None:
    """Print the number of rows and dates in one split."""
    first_time = data[TIME_COLUMN].min()
    last_time = data[TIME_COLUMN].max()
    print(f"{split_name}: {len(data):,} rows ({first_time} to {last_time})")


def save_splits(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save the scaled data sets as separate CSV files."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(OUTPUT_FOLDER / "train_scaled.csv", index=False)
    validation_data.to_csv(OUTPUT_FOLDER / "validation_scaled.csv", index=False)
    test_data.to_csv(OUTPUT_FOLDER / "test_scaled.csv", index=False)


def main() -> None:
    # Step 1: load the full cleaned data set.
    data = load_data()

    # Step 2: split it by time. Do not shuffle time-series data.
    training_data, validation_data, test_data = split_by_time(data)

    # Step 3: fit one scaler on training data, then reuse it for all splits.
    training_scaled, validation_scaled, test_scaled, scaler = scale_numeric_columns(
        training_data, validation_data, test_data
    )

    print_split_details("Training (2020-2022)", training_scaled)
    print_split_details("Validation (2023)", validation_scaled)
    print_split_details("Test (2024+)", test_scaled)
    print(f"\nTraining-only averages for {NUMERIC_COLUMNS}: {scaler.mean_}")
    print(f"Training-only standard deviations: {scaler.scale_}")

    # Step 4: save the results.
    save_splits(training_scaled, validation_scaled, test_scaled)
    print(f"\nSaved scaled CSV files to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
