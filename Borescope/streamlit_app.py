from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from borescope_analysis import (
    RAW_DIR,
    TYPE_MAP_FILE,
    add_type_column,
    build_top50,
    build_top50_summary,
    load_all_raw_data,
)


BASE_DIR = Path(__file__).resolve().parent


def resolve_metric_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return the revenue and units column names present in the DataFrame."""
    revenue_col = None
    units_col = None

    if "ASIN Revenue" in df.columns:
        revenue_col = "ASIN Revenue"
    elif "Est. Monthly Retail Rev" in df.columns:
        revenue_col = "Est. Monthly Retail Rev"

    if "ASIN Sales" in df.columns:
        units_col = "ASIN Sales"
    elif "Est. Monthly Units Sold" in df.columns:
        units_col = "Est. Monthly Units Sold"

    return revenue_col, units_col


def load_typed_data() -> pd.DataFrame:
    """Load raw CSVs, filter to borescopes, enforce numeric types, and attach Type mapping."""
    df_raw = load_all_raw_data(RAW_DIR)
    if "Subcategory" in df_raw.columns:
        df_raw = df_raw[df_raw["Subcategory"] == "Borescopes"].copy()

    numeric_cols = ["Price", "ASIN Revenue", "ASIN Sales", "Review Count", "Reviews Rating"]
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    return add_type_column(df_raw, TYPE_MAP_FILE)


def apply_filters(df: pd.DataFrame):
    """Render sidebar controls and return a filtered DataFrame plus selections."""
    brands = sorted(df["Brand"].dropna().astype(str).unique())
    price_min, price_max = float(df["Price"].min()), float(df["Price"].max())
    rating_min, rating_max = float(df["Reviews Rating"].min()), float(df["Reviews Rating"].max())

    # Initialize session defaults
    st.session_state.setdefault("brand_filter", "All brands")
    st.session_state.setdefault("price_range", (price_min, price_max))
    st.session_state.setdefault("rating_range", (rating_min, rating_max))

    st.sidebar.header("Filters")
    show_all = st.sidebar.checkbox(
        "View all data",
        value=False,
        help="Ignore the numeric filters to view the full dataset.",
    )
    if st.sidebar.button("Reset all filters"):
        st.session_state.brand_filter = "All brands"
        st.session_state.price_range = (price_min, price_max)
        st.session_state.rating_range = (rating_min, rating_max)
        st.rerun()

    brand = st.sidebar.selectbox(
        "Brand", ["All brands"] + brands, index=0, key="brand_filter", help="Choose a single brand or all brands"
    )
    price_range = st.sidebar.slider(
        "Price range",
        min_value=price_min,
        max_value=price_max,
        value=st.session_state.price_range,
        format="$%0.2f",
        key="price_range",
    )
    rating_range = st.sidebar.slider(
        "Rating range",
        min_value=rating_min,
        max_value=rating_max,
        value=st.session_state.rating_range,
        step=0.1,
        key="rating_range",
    )

    mask = pd.Series(True, index=df.index)
    if not show_all:
        # Allow rows with missing numeric values to pass through so counts match "view all"
        mask &= df["Price"].between(price_range[0], price_range[1]) | df["Price"].isna()
        mask &= df["Reviews Rating"].between(rating_range[0], rating_range[1]) | df["Reviews Rating"].isna()
        if brand != "All brands":
            mask &= df["Brand"].astype(str) == brand
    elif brand != "All brands":
        # Allow viewing all rows while still drilling into a brand when selected.
        mask &= df["Brand"].astype(str) == brand

    return df[mask].copy(), brand, show_all


def type_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and units by Type for pie charts."""
    revenue_col, units_col = resolve_metric_columns(df)
    if not revenue_col or not units_col:
        missing = ["ASIN Revenue", "ASIN Sales", "Est. Monthly Retail Rev", "Est. Monthly Units Sold"]
        raise KeyError(f"Required revenue/units columns not found. Checked: {missing}")

    data = df.copy()
    data["Type"] = data["Type"].fillna("Unmapped")
    grouped = (
        data.groupby("Type", dropna=False)
        .agg(Revenue=(revenue_col, "sum"), Units=(units_col, "sum"))
        .reset_index()
    )
    return grouped


def brand_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue and units by Brand for share pies."""
    revenue_col, units_col = resolve_metric_columns(df)
    if not revenue_col or not units_col:
        missing = ["ASIN Revenue", "ASIN Sales", "Est. Monthly Retail Rev", "Est. Monthly Units Sold"]
        raise KeyError(f"Required revenue/units columns not found. Checked: {missing}")

    data = df.copy()
    data["Brand"] = data["Brand"].fillna("Unknown Brand")
    grouped = (
        data.groupby("Brand", dropna=False)
        .agg(Revenue=(revenue_col, "sum"), Units=(units_col, "sum"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )
    return grouped


def pie_chart(data: pd.DataFrame, label_col: str, value_col: str, title: str):
    if data.empty or data[value_col].sum() == 0:
        st.info(f"No data to display for {title}.")
        return

    chart = (
        alt.Chart(data)
        .mark_arc(innerRadius=60)
        .encode(
            theta=alt.Theta(field=value_col, type="quantitative"),
            color=alt.Color(f"{label_col}:N", legend=alt.Legend(title=label_col)),
            tooltip=[label_col, alt.Tooltip(value_col, format=",.0f")],
        )
        .properties(title=title)
    )
    st.altair_chart(chart, use_container_width=True)


def layout_top50_sections(df_filtered: pd.DataFrame):
    if df_filtered.empty:
        st.warning("No data after filters. Try expanding your ranges or resetting filters.")
        return

    top50_rev = build_top50(df_filtered, "ASIN Revenue")
    top50_units = build_top50(df_filtered, "ASIN Sales")
    top50_summary = build_top50_summary(top50_rev)

    st.subheader("Top 50 (Revenue)")
    st.dataframe(top50_rev, use_container_width=True)

    st.subheader("Top 50 (Units)")
    st.dataframe(top50_units, use_container_width=True)

    st.subheader("Top 50 Summary")
    st.dataframe(top50_summary, use_container_width=True)

    st.subheader("Type Share (Top 50 only)")
    col1, col2 = st.columns(2)
    top50_rev_shares = type_shares(top50_rev)
    top50_units_shares = type_shares(top50_units)
    with col1:
        pie_chart(top50_rev_shares, "Type", "Revenue", "Top 50 Revenue Share by Type")
    with col2:
        pie_chart(top50_units_shares, "Type", "Units", "Top 50 Units Share by Type")


def top_brands_revenue_chart(df: pd.DataFrame):
    """Bar chart of top 10 brands by monthly revenue."""
    revenue_col, _ = resolve_metric_columns(df)
    if not revenue_col:
        st.info("Revenue column missing; cannot build brand ranking.")
        return

    data = df.copy()
    data["Brand"] = data["Brand"].fillna("Unknown Brand")

    grouped = (
        data.groupby("Brand", dropna=False)
        .agg(MonthlyRevenue=(revenue_col, "sum"))
        .reset_index()
        .sort_values("MonthlyRevenue", ascending=False)
        .head(10)
    )

    if grouped.empty:
        st.info("No data available for brand revenue ranking.")
        return

    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            y=alt.Y("Brand:N", sort="-x", title="Brand"),
            x=alt.X("MonthlyRevenue:Q", title="Monthly Revenue ($)", axis=alt.Axis(format="$,.0f")),
            tooltip=["Brand", alt.Tooltip("MonthlyRevenue:Q", title="Monthly Revenue", format="$,.0f")],
        )
        .properties(height=400)
    )
    st.altair_chart(chart, use_container_width=True)


def render_brand_ranking(df: pd.DataFrame, brand: str):
    """Show all ASINs for a brand ranked by monthly revenue/units."""
    if brand == "All brands":
        return

    brand_df = df[df["Brand"].astype(str) == brand].copy()
    if brand_df.empty:
        st.info(f"No ASINs found for brand '{brand}'.")
        return

    revenue_col, units_col = resolve_metric_columns(brand_df)
    if not revenue_col or not units_col:
        st.info("Revenue/units columns missing; cannot rank ASINs.")
        return

    brand_df["Rank by Monthly Rev"] = (
        brand_df[revenue_col].rank(method="dense", ascending=False, na_option="bottom").fillna(0).astype(int)
    )
    brand_df["Rank by Monthly Units"] = (
        brand_df[units_col].rank(method="dense", ascending=False, na_option="bottom").fillna(0).astype(int)
    )

    brand_df = brand_df.sort_values(revenue_col, ascending=False)

    display_cols = [
        "Rank by Monthly Rev",
        "Rank by Monthly Units",
        "ASIN",
        "Title",
        "Type",
        "Price",
        revenue_col,
        units_col,
        "Review Count",
        "Reviews Rating",
        "URL",
    ]
    available_cols = [col for col in display_cols if col in brand_df.columns]

    st.subheader(f"ASIN Ranking for {brand}")
    st.dataframe(brand_df[available_cols], use_container_width=True, height=500)


def boxplot_section(df: pd.DataFrame):
    """Render boxplots for price, revenue, and units distributions by Type."""
    if df.empty:
        st.info("No data to plot boxplots.")
        return

    revenue_col, units_col = resolve_metric_columns(df)
    if not revenue_col or not units_col:
        st.info("Revenue/units columns missing; cannot render boxplots.")
        return

    data = df.copy()
    data["Type"] = data["Type"].fillna("Unmapped")

    def build_box_chart(value_col: str, title: str, fmt: str = ",.0f"):
        return (
            alt.Chart(data)
            .mark_boxplot()
            .encode(
                x=alt.X("Type:N", title="Type"),
                y=alt.Y(f"{value_col}:Q", title=title),
                color=alt.Color("Type:N", legend=None),
                tooltip=["Type", alt.Tooltip(value_col, format=fmt)],
            )
            .properties(height=300)
        )

    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(build_box_chart("Price", "Price ($)", "$,.2f"), use_container_width=True)
    with col2:
        st.altair_chart(build_box_chart(units_col, "Monthly Units Sold"), use_container_width=True)

    st.altair_chart(build_box_chart(revenue_col, "Monthly Revenue ($)", "$,.0f"), use_container_width=True)


def main():
    st.title("Borescope Market Dashboard")
    st.caption(f"Data folder: {BASE_DIR}")

    df = load_typed_data()
    df_filtered, selected_brand, view_all = apply_filters(df)

    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            white-space: nowrap;
            overflow: visible;
            text-overflow: clip;
        }
        [data-testid="stMetricLabel"] { white-space: nowrap; }
        div[data-testid="metric-container"] { min-width: 180px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    revenue_col, units_col = resolve_metric_columns(df_filtered)
    if not revenue_col or not units_col:
        st.info("Revenue/units columns missing; cannot render summary.")
        return

    # Summary metrics
    total_rev = df_filtered[revenue_col].sum()
    total_units = df_filtered[units_col].sum()
    avg_rating = df_filtered["Reviews Rating"].mean()
    brand_count = df_filtered["Brand"].dropna().nunique()
    asin_count = df_filtered["ASIN"].dropna().nunique()

    st.subheader("Summary")
    row_top = st.columns(3)
    row_bottom = st.columns(2)
    row_top[0].metric("Total Units Sold", f"{total_units:,.0f}")
    row_top[1].metric("Total Revenue", f"${total_rev:,.0f}")
    row_top[2].metric("Total Brands", f"{brand_count:,}")
    row_bottom[0].metric("Total ASINs", f"{asin_count:,}")
    row_bottom[1].metric("Avg Rating", f"{avg_rating:.2f}" if pd.notna(avg_rating) else "N/A")

    st.divider()

    st.subheader("Top 10 Brands by Revenue")
    top_brands_revenue_chart(df_filtered)

    st.divider()

    st.subheader("Market Share by Brand (Filtered)")
    shares = brand_shares(df_filtered)
    col_share1, col_share2 = st.columns(2)
    with col_share1:
        pie_chart(shares, "Brand", "Revenue", "Monthly Revenue Share by Brand")
    with col_share2:
        pie_chart(shares, "Brand", "Units", "Monthly Units Share by Brand")

    st.divider()

    layout_top50_sections(df_filtered)

    if selected_brand != "All brands":
        st.divider()
        render_brand_ranking(df, selected_brand if not view_all else selected_brand)

    st.divider()

    st.subheader("Distribution Insights")
    boxplot_section(df_filtered)

    st.divider()

    if st.button("View all filtered data"):
        st.dataframe(df_filtered, use_container_width=True, height=500)


if __name__ == "__main__":
    main()
