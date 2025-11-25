import pandas as pd
from pathlib import Path

# ======== PATHS & SETTINGS ========
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"  # this is "/Users/sumiaoc/Desktop/Thermal Imager/raw_data"
OUTPUT_FILE = BASE_DIR / "Thermal Imager Top 50_auto.xlsx"

TOP_N = 50
SUBCATEGORY_FILTER = "Thermal Imagers"  # change if your Subcategory text is different
# ==================================


def load_all_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Load and combine all CSV files in raw_data folder."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    dfs = []
    for f in csv_files:
        print(f"Loading: {f.name}")
        df = pd.read_csv(f)
        df["source_file"] = f.name  # optional: track which file it came from
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape from {len(csv_files)} files:", combined.shape)
    return combined


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to subcategory and coerce numeric columns once."""
    if "Subcategory" in df.columns:
        df = df[df["Subcategory"] == SUBCATEGORY_FILTER].copy()
        print("After Subcategory filter:", df.shape)
    else:
        print("Warning: 'Subcategory' column not found, using all rows.")
        df = df.copy()

    numeric_cols = ["Price", "ASIN Revenue", "ASIN Sales", "Review Count", "Reviews Rating"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_top_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Build a Top N table sorted by the requested metric."""
    if metric_col not in df.columns:
        raise ValueError(f"Column '{metric_col}' not found in the raw data. Check CSV headers.")

    sorted_df = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    top = sorted_df.head(TOP_N).copy()

    return pd.DataFrame({
        "Ranking": range(1, len(top) + 1),
        "ASIN": top["ASIN"],
        "Product Name": top["Title"],
        "Brand": top["Brand"].astype(str).str.lower(),  # optional: lowercase brand
        "Price": top["Price"],
        "Est. Monthly Retail Rev": top["ASIN Revenue"],
        "Est. Monthly Units Sold": top["ASIN Sales"],
        "# of Reviews": top["Review Count"],
        "Avg. Rating": top["Reviews Rating"],
        "Column2": top["URL"],    # helper column you had in your template
        "Link": top["URL"],       # clickable URL column
    })


def save_tables(tables: list[tuple[str, str, pd.DataFrame]], outfile: Path):
    """Save multiple tables, each with its own sheet title row."""
    with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
        for sheet_name, title, output_df in tables:
            title_row = pd.DataFrame(
                [[title] + [None] * (output_df.shape[1] - 1)],
                columns=output_df.columns
            )
            title_row.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            output_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=1)

    print(f"\nSaved {len(tables)} tables to: {outfile}")


def main():
    # Step 1: load all raw CSVs from /raw_data
    df_raw = load_all_raw_data(RAW_DIR)

    # Step 2: prep dataframe and build both rankings
    df_prepped = prepare_base_dataframe(df_raw)
    revenue_top = build_top_table(df_prepped, "ASIN Revenue")
    units_top = build_top_table(df_prepped, "ASIN Sales")

    # Step 3: save both sections into the Excel file
    tables_to_save = [
        ("Top 50 Revenue", "Rank By Monthly Revenue - ONLY", revenue_top),
        ("Top 50 Units", "Rank By Monthly Units - ONLY", units_top),
    ]
    save_tables(tables_to_save, OUTPUT_FILE)


if __name__ == "__main__":
    main()
