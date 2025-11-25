from pathlib import Path

import pandas as pd

# Number of rows to keep for manual type labeling
TOP_N = 200

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
OUTPUT_FILE = BASE_DIR / "type_mapping_template.xlsx"


def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all CSV files in the raw_data directory."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = [pd.read_csv(csv_file) for csv_file in csv_files]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = load_raw_data(RAW_DIR)

    # Filter to borescopes only if the column exists
    if "Subcategory" in df.columns:
        df = df[df["Subcategory"] == "Borescopes"].copy()

    if "ASIN Revenue" not in df.columns:
        raise ValueError("Missing required column 'ASIN Revenue' in raw data.")

    required_cols = ["ASIN", "Title", "Brand", "URL"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    df["ASIN Revenue"] = pd.to_numeric(df["ASIN Revenue"], errors="coerce")
    df = df.sort_values("ASIN Revenue", ascending=False).head(TOP_N)

    template = df[required_cols].copy()
    template["Type"] = ""  # Placeholder for manual labeling

    template.to_excel(OUTPUT_FILE, index=False)
    print(f"Type mapping template saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
