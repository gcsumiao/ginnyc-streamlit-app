from pathlib import Path
import re

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
TYPE_MAP_FILE = BASE_DIR / "type_mapping.xlsx"
OUTPUT_FILE = BASE_DIR / "Borescope_Market_Analysis.xlsx"
TOP_N = 50


def load_all_raw_data(raw_dir: Path) -> pd.DataFrame:
    """
    Load all CSV files in raw_dir and concatenate into one DataFrame.
    Add a 'source_file' column with the original filename.
    """
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["source_file"] = csv_file.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def add_type_column(df: pd.DataFrame, type_map_file: Path) -> pd.DataFrame:
    """
    Left-join a manual type mapping (from type_mapping.xlsx) onto df by ASIN.
    """
    if not type_map_file.exists():
        raise FileNotFoundError(
            f"Type mapping file not found: {type_map_file}. Please create it first."
        )

    type_map = pd.read_excel(type_map_file)
    if "ASIN" not in type_map.columns or "Type" not in type_map.columns:
        raise ValueError("type_mapping.xlsx must contain 'ASIN' and 'Type' columns.")

    df = df.copy()
    df["ASIN"] = df["ASIN"].astype(str)
    type_map["ASIN"] = type_map["ASIN"].astype(str)

    merged = df.merge(type_map[["ASIN", "Type"]], on="ASIN", how="left")

    missing_type = merged["Type"].isna().sum()
    if missing_type > 0:
        print(f"Warning: {missing_type} rows are missing a mapped Type.")

    return merged


def ensure_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def sanitize_sheet_name(name: str) -> str:
    """Remove invalid Excel sheet characters and trim length."""
    cleaned = re.sub(r"[\\/*?:\\[\\]]", " ", str(name))
    cleaned = cleaned.strip() or "Sheet"
    return cleaned[:31]


def build_top50(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    ensure_required_columns(
        df,
        ["ASIN", "Title", "Brand", "URL", "ASIN Revenue", "ASIN Sales", "Review Count", "Reviews Rating"],
    )
    if sort_by not in df.columns:
        raise ValueError(f"Missing required column for ranking: {sort_by}")

    top = df.sort_values(sort_by, ascending=False).head(TOP_N).copy()

    top50_output = pd.DataFrame(
        {
            "Ranking": range(1, len(top) + 1),
            "ASIN": top["ASIN"],
            "Product Name": top["Title"],
            "Brand": top["Brand"].astype(str).str.lower(),
            "Type": top.get("Type", pd.Series([""] * len(top))),
            "Price": top["Price"],
            "Est. Monthly Retail Rev": top["ASIN Revenue"],
            "Est. Monthly Units Sold": top["ASIN Sales"],
            "# of Reviews": top["Review Count"],
            "Avg. Rating": top["Reviews Rating"],
            "Column2": top["URL"],
            "Link": top["URL"],
        }
    )
    return top50_output


def build_top50_summary(top50_df: pd.DataFrame) -> pd.DataFrame:
    units_total = top50_df["Est. Monthly Units Sold"].sum()
    revenue_total = top50_df["Est. Monthly Retail Rev"].sum()

    summary = (
        top50_df.groupby("Type", dropna=False)
        .agg(
            Avg_Price=("Price", "mean"),
            Quantity_Mo=("Est. Monthly Units Sold", "sum"),
            Revenue_Mo=("Est. Monthly Retail Rev", "sum"),
        )
        .reset_index()
        .rename(columns={"Type": "Type"})
    )

    summary["Qty by %"] = summary["Quantity_Mo"] / units_total if units_total else 0
    summary["Revenue by %"] = summary["Revenue_Mo"] / revenue_total if revenue_total else 0

    total_row = {
        "Type": "Total",
        "Avg_Price": summary["Avg_Price"].mean(),
        "Quantity_Mo": units_total,
        "Revenue_Mo": revenue_total,
        "Qty by %": 1.0 if units_total else 0,
        "Revenue by %": 1.0 if revenue_total else 0,
    }

    summary = pd.concat([summary, pd.DataFrame([total_row])], ignore_index=True)
    summary = summary.rename(
        columns={
            "Avg_Price": "Avg Price",
            "Quantity_Mo": "Quantity/Mo",
            "Revenue_Mo": "Revenue/Mo",
        }
    )
    return summary


def build_brand_summary(df: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(
        df,
        ["ASIN", "Brand", "ASIN Revenue", "ASIN Sales", "Review Count", "Reviews Rating"],
    )

    df = df.copy()
    df["rating_weight"] = df["Reviews Rating"] * df["Review Count"]

    grouped = df.groupby("Brand", dropna=False).agg(
        listings=("ASIN", "nunique"),
        monthly_rev=("ASIN Revenue", "sum"),
        monthly_units=("ASIN Sales", "sum"),
        total_reviews=("Review Count", "sum"),
        rating_weight=("rating_weight", "sum"),
    )

    grouped["Estimated 12mo Revenue"] = grouped["monthly_rev"] * 12
    grouped["Estimated 12mo Units"] = grouped["monthly_units"] * 12
    grouped["Avg Rating"] = grouped.apply(
        lambda row: row["rating_weight"] / row["total_reviews"] if row["total_reviews"] else float("nan"),
        axis=1,
    )
    grouped["Price Per Unit"] = grouped.apply(
        lambda row: row["monthly_rev"] / row["monthly_units"] if row["monthly_units"] else float("nan"),
        axis=1,
    )

    total_rev = grouped["monthly_rev"].sum()
    total_units = grouped["monthly_units"].sum()

    grouped["Monthly Rev Market Share %"] = grouped["monthly_rev"] / total_rev if total_rev else 0
    grouped["Monthly Unit Market Share %"] = grouped["monthly_units"] / total_units if total_units else 0

    grouped = grouped.reset_index().rename(
        columns={
            "Brand": "Brand",
            "listings": "# of Listings",
            "monthly_rev": "Monthly Rev",
            "monthly_units": "Monthly Units",
            "total_reviews": "Total Reviews",
        }
    )

    total_row = {
        "Brand": "Total",
        "# of Listings": grouped["# of Listings"].sum(),
        "Monthly Rev": total_rev,
        "Estimated 12mo Revenue": grouped["Estimated 12mo Revenue"].sum(),
        "Monthly Units": total_units,
        "Estimated 12mo Units": grouped["Estimated 12mo Units"].sum(),
        "Monthly Rev Market Share %": 1.0 if total_rev else 0,
        "Monthly Unit Market Share %": 1.0 if total_units else 0,
        "Price Per Unit": total_rev / total_units if total_units else float("nan"),
        "Total Reviews": grouped["Total Reviews"].sum(),
        "Avg Rating": (
            grouped["Avg Rating"].mul(grouped["Total Reviews"]).sum() / grouped["Total Reviews"].sum()
            if grouped["Total Reviews"].sum()
            else float("nan")
        ),
    }

    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)
    return grouped


def write_brand_sheets(writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
    brands = df["Brand"].dropna().unique()
    for brand in brands:
        brand_df = df[df["Brand"] == brand].copy()
        brand_df = brand_df.sort_values("ASIN Revenue", ascending=False)

        rank_rev = brand_df["ASIN Revenue"].rank(method="dense", ascending=False, na_option="bottom")
        rank_units = brand_df["ASIN Sales"].rank(method="dense", ascending=False, na_option="bottom")

        brand_df["Rank by Monthly Rev"] = rank_rev.fillna(0).astype(int)
        brand_df["Rank by Monthly Units"] = rank_units.fillna(0).astype(int)

        output_cols = [
            "ASIN",
            "Title",
            "Type",
            "Price",
            "ASIN Revenue",
            "ASIN Sales",
            "Review Count",
            "Reviews Rating",
            "URL",
            "Rank by Monthly Rev",
            "Rank by Monthly Units",
        ]

        missing = [col for col in output_cols if col not in brand_df.columns]
        if missing:
            raise ValueError(f"Missing required columns for brand sheet: {missing}")

        sheet_name = sanitize_sheet_name(brand)
        brand_df.rename(
            columns={
                "Title": "Product Name",
                "ASIN Revenue": "Est. Monthly Retail Rev",
                "ASIN Sales": "Est. Monthly Units Sold",
                "Review Count": "# of Reviews",
                "Reviews Rating": "Avg. Rating",
                "URL": "Link",
            },
            inplace=True,
        )

        brand_df[
            [
                "ASIN",
                "Product Name",
                "Type",
                "Price",
                "Est. Monthly Retail Rev",
                "Est. Monthly Units Sold",
                "# of Reviews",
                "Avg. Rating",
                "Link",
                "Rank by Monthly Rev",
                "Rank by Monthly Units",
            ]
        ].to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    df_raw = load_all_raw_data(RAW_DIR)

    if "Subcategory" in df_raw.columns:
        df_raw = df_raw[df_raw["Subcategory"] == "Borescopes"].copy()

    numeric_cols = ["Price", "ASIN Revenue", "ASIN Sales", "Review Count", "Reviews Rating"]
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_typed = add_type_column(df_raw, TYPE_MAP_FILE)

    top50_rev = build_top50(df_typed, "ASIN Revenue")
    top50_units = build_top50(df_typed, "ASIN Sales")
    top50_summary = build_top50_summary(top50_rev)
    brand_summary = build_brand_summary(df_typed)

    # Revenue-focused table
    revenue_columns = [
        "Brand",
        "# of Listings",
        "Monthly Rev",
        "Estimated 12mo Revenue",
        "Monthly Rev Market Share %",
        "Price Per Unit",
        "Total Reviews",
        "Avg Rating",
    ]

    # Units-focused table
    units_columns = [
        "Brand",
        "# of Listings",
        "Monthly Units",
        "Estimated 12mo Units",
        "Monthly Unit Market Share %",
        "Avg Rating",
    ]

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        # Top 50 Revenue sheet with title row
        top50_rev.to_excel(writer, sheet_name="Top 50 Revenue", index=False, startrow=1)
        sheet = writer.sheets["Top 50 Revenue"]
        sheet.cell(row=1, column=1, value="Rank By Monthly Revenue - ONLY")

        # Top 50 Units sheet with title row
        top50_units.to_excel(writer, sheet_name="Top 50 Units", index=False, startrow=1)
        units_sheet = writer.sheets["Top 50 Units"]
        units_sheet.cell(row=1, column=1, value="Rank By Monthly Units - ONLY")

        # Top 50 Summary sheet
        top50_summary.to_excel(writer, sheet_name="Top 50 Summary", index=False)

        # Summary sheet
        brand_summary[revenue_columns].to_excel(writer, sheet_name="Summary", index=False, startrow=1)
        summary_sheet = writer.sheets["Summary"]
        summary_sheet.cell(row=1, column=1, value="Monthly Summaries - Revenue")

        # Units table below revenue table with a blank row
        start_row_units = len(brand_summary) + 4
        brand_summary[units_columns].to_excel(
            writer, sheet_name="Summary", index=False, startrow=start_row_units
        )
        summary_sheet.cell(row=start_row_units, column=1, value="Monthly Summary - Units")

        # Brand detail sheets
        write_brand_sheets(writer, df_typed)

    print(f"Analysis workbook created at: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
