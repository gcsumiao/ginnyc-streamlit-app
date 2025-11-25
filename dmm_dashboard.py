from __future__ import annotations

import hashlib
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------- CONFIG ----------
RAW_DATA_DIR = Path(__file__).parent / "raw_data"
CSV_PATTERN = "*.csv"
NUM_COLS = [
    "Price",
    "BSR",
    "ASIN Sales",
    "ASIN Revenue",
    "Parent Level Sales",
    "Parent Level Revenue",
    "Review Count",
    "Reviews Rating",
    "Sales Trend (90 days) (%)",
    "Price Trend (90 days) (%)",
    "Sales Year Over Year (%)",
]
REQUIRED_COLS = ["ASIN", "Title", "Category", "Subcategory", "Price", "ASIN Sales", "ASIN Revenue"]
CACHE_VERSION = "2025-02-21"


# ---------- HELPERS ----------
def _list_data_files() -> list[Path]:
    if not RAW_DATA_DIR.exists():
        return []
    return sorted(RAW_DATA_DIR.glob(CSV_PATTERN))


def _cache_key(files: list[Path]) -> str:
    fingerprint = [(f.name, f.stat().st_mtime_ns, f.stat().st_size) for f in files]
    raw = repr((CACHE_VERSION, fingerprint)).encode()
    return hashlib.sha1(raw).hexdigest()


# ---------- DATA PREP ----------
@st.cache_data(show_spinner=False)
def load_and_prepare_data(cache_token: str) -> dict[str, pd.DataFrame]:
    files = _list_data_files()
    if not files:
        st.error(f"No CSV files found in {RAW_DATA_DIR} (pattern: {CSV_PATTERN})")
        st.stop()

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["ASIN"])

    missing = [c for c in REQUIRED_COLS if c not in df_all.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Category/Subcategory filters
    auto = df_all[df_all["Category"].astype(str).str.contains("Automotive", case=False, na=False)]
    auto = auto[auto["Subcategory"].astype(str).str.contains("Multimeter|Analyzer", case=False, na=False)]

    # Numeric cleaning
    for c in NUM_COLS:
        if c in auto.columns:
            auto[c] = pd.to_numeric(auto[c], errors="coerce")

    # Title flags (vectorized)
    title_lower = auto["Title"].fillna("").str.lower()
    auto["is_multimeter"] = title_lower.str.contains("multimeter")
    auto["is_analyzer"] = title_lower.str.contains("analyzer|analyser")

    screen_pattern = re.compile(
        "large screen|large display|big screen|color screen|hd screen|hd display|lcd display|full screen|smart display",
        re.IGNORECASE,
    )
    auto["is_large_screen_like"] = title_lower.str.contains(screen_pattern)
        # Color / advanced display keywords
    color_display_pattern = re.compile(
        "color lcd|color screen|full color display|tft display|tft lcd|ips display|oled display",
        re.IGNORECASE,
    )
    auto["is_color_display"] = title_lower.str.contains(color_display_pattern)

    auto["is_rechargeable"] = title_lower.str.contains("rechargeable|type-c|usb c|usb-c")
    auto["is_automotive_targeted"] = title_lower.str.contains("car|automotive|vehicle|battery tester")

    # Product type classification (vectorized)
    auto["product_type"] = np.select(
        [auto["is_multimeter"] & auto["is_analyzer"], auto["is_multimeter"], auto["is_analyzer"]],
        ["Multimeter + Analyzer", "Multimeter", "Analyzer"],
        default="Other",
    )

    # Brand cleaning + INNOVA flag
    if "Brand" in auto.columns:
        auto["brand_clean"] = auto["Brand"].astype(str).str.strip().str.upper()
    else:
        auto["brand_clean"] = "UNKNOWN"
    auto["is_innova"] = auto["brand_clean"].str.contains("INNOVA", case=False, na=False)

    # Core dataset (basic quality filters)
    core = auto.copy()
    if "ASIN Sales" in core.columns:
        core = core[core["ASIN Sales"].notna()]
    if "ASIN Revenue" in core.columns:
        core = core[core["ASIN Revenue"].notna()]
    if "Price" in core.columns:
        core = core[core["Price"] > 0]

    # Overall KPIs
    total_revenue = core["ASIN Revenue"].sum()
    total_sales = core["ASIN Sales"].sum()
    asin_count = core["ASIN"].nunique()
    avg_price = core["Price"].mean()
    median_price = core["Price"].median()
    avg_rating = core["Reviews Rating"].mean() if "Reviews Rating" in core.columns else np.nan
    median_rating = core["Reviews Rating"].median() if "Reviews Rating" in core.columns else np.nan

    kpi_overview = pd.DataFrame(
        [
            {
                "metric": "Overall market (last 30 days)",
                "total_revenue": total_revenue,
                "total_sales": total_sales,
                "asin_count": asin_count,
                "avg_price": avg_price,
                "median_price": median_price,
                "avg_rating": avg_rating,
                "median_rating": median_rating,
            }
        ]
    )

    # Brand-level summary
    brand_summary = (
        core.groupby("brand_clean", as_index=False)
        .agg(
            total_revenue=("ASIN Revenue", "sum"),
            total_sales=("ASIN Sales", "sum"),
            asin_count=("ASIN", "nunique"),
            avg_price=("Price", "mean"),
            avg_rating=("Reviews Rating", "mean"),
            total_reviews=("Review Count", "sum"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    # Product type summary
    product_type_summary = (
        core.groupby("product_type", as_index=False)
        .agg(
            total_revenue=("ASIN Revenue", "sum"),
            total_sales=("ASIN Sales", "sum"),
            asin_count=("ASIN", "nunique"),
            avg_price=("Price", "mean"),
            avg_rating=("Reviews Rating", "mean"),
            total_reviews=("Review Count", "sum"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    # Price tiers
    price_bins = [0, 30, 60, 100, 200, np.inf]
    price_labels = ["<$30", "$30–59", "$60–99", "$100–199", "$200+"]
    core["price_tier"] = pd.cut(core["Price"], bins=price_bins, labels=price_labels, right=False)

    # Large-screen vs non-large-screen price and sales differences
    large_screen_summary = (
        core.groupby("is_large_screen_like", as_index=False)
        .agg(
            total_revenue=("ASIN Revenue", "sum"),
            total_sales=("ASIN Sales", "sum"),
            avg_price=("Price", "mean"),
            avg_rating=("Reviews Rating", "mean"),
            asin_count=("ASIN", "nunique"),
        )
    )
    # Add readable labels for True/False
    large_screen_summary["segment"] = large_screen_summary["is_large_screen_like"].map(
        {True: "Large-screen-like", False: "Non-large-screen"}
    )

    # Color display vs non-color display premium
    color_display_summary = (
        core.groupby("is_color_display", as_index=False)
        .agg(
            total_revenue=("ASIN Revenue", "sum"),
            total_sales=("ASIN Sales", "sum"),
            avg_price=("Price", "mean"),
            avg_rating=("Reviews Rating", "mean"),
            asin_count=("ASIN", "nunique"),
        )
    )
    color_display_summary["segment"] = color_display_summary["is_color_display"].map(
        {True: "Color / advanced display", False: "Non-color display"}
    )

    battery_category_summary = (
        core.groupby("is_rechargeable", as_index=False)
        .agg(
            total_revenue=("ASIN Revenue", "sum"),
            total_sales=("ASIN Sales", "sum"),
            avg_price=("Price", "mean"),
            avg_rating=("Reviews Rating", "mean"),
            asin_count=("ASIN", "nunique"),
        )
    )
    battery_category_summary["segment"] = battery_category_summary["is_rechargeable"].map(
        {True: "Rechargeable", False: "Non-rechargeable / unspecified"}
    )

    price_tier_summary = (
        core.groupby("price_tier", as_index=False, observed=False)
        .agg(
            total_revenue=("ASIN Revenue", "sum"),
            total_sales=("ASIN Sales", "sum"),
            asin_count=("ASIN", "nunique"),
            avg_price=("Price", "mean"),
            avg_rating=("Reviews Rating", "mean"),
            total_reviews=("Review Count", "sum"),
        )
        .sort_values("total_revenue", ascending=False)
    )

    # Top ASINs table
    top_asins = (
        core.sort_values("ASIN Revenue", ascending=False)
        .loc[
            :,
            [
                "ASIN",
                "Title",
                "brand_clean",
                "product_type",
                "Price",
                "ASIN Sales",
                "ASIN Revenue",
                "Review Count",
                "Reviews Rating",
                "is_large_screen_like",
                "is_rechargeable",
                "is_automotive_targeted",
            ],
        ]
    )

    return {
        "core": core,
        "brand_summary": brand_summary,
        "product_type_summary": product_type_summary,
        "price_tier_summary": price_tier_summary,
        "top_asins": top_asins,
        "kpi_overview": kpi_overview,
        "large_screen_summary": large_screen_summary,
        "color_display_summary": color_display_summary,
        "battery_category_summary": battery_category_summary,
    }


def render_pie_chart(df: pd.DataFrame, label_col: str, value_col: str, value_label: str) -> None:
    if df.empty or df[value_col].sum() <= 0:
        st.write("No data for current filters.")
        return

    pie_df = df[[label_col, value_col]].dropna(subset=[label_col]).copy()
    pie_df[label_col] = pie_df[label_col].astype(str)
    pie_df["share_pct"] = pie_df[value_col] / pie_df[value_col].sum() * 100

    chart = (
        alt.Chart(pie_df)
        .mark_arc(innerRadius=40)
        .encode(
            theta=alt.Theta(field=value_col, type="quantitative"),
            color=alt.Color(field=label_col, type="nominal", legend=alt.Legend(title=None)),
            tooltip=[
                alt.Tooltip(field=label_col, type="nominal", title=label_col.replace("_", " ").title()),
                alt.Tooltip(field=value_col, type="quantitative", title=value_label, format=",.0f"),
                alt.Tooltip(field="share_pct", type="quantitative", title="Share (%)", format=".1f"),
            ],
        )
    )
    st.altair_chart(chart, use_container_width=True)


# ---------- DASHBOARD LAYOUT ----------
st.set_page_config(page_title="Automotive DMM Market Dashboard", layout="wide")
st.title("Automotive DMMs – Amazon Market Dashboard")
st.caption("Based on Helium10 BlackBox data (last 30 days)")

files = _list_data_files()
data = load_and_prepare_data(_cache_key(files))
core = data["core"]
brand_summary = data["brand_summary"]
product_type_summary = data["product_type_summary"]
price_tier_summary = data["price_tier_summary"]
top_asins = data["top_asins"]
kpi_overview = data["kpi_overview"]
large_screen_summary = data["large_screen_summary"]
color_display_summary = data["color_display_summary"]
battery_category_summary = data["battery_category_summary"]


# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("Filters")

DEFAULT_FILTERS = {
    "brand": "All",
    "ptype": "All",
    "ptier": "All",
    "only_auto": False,
    "only_large": False,
    "only_recharge": False,
}

# Refresh button to reload data files and cache
if st.sidebar.button("Refresh data (reload files)"):
    load_and_prepare_data.clear()
    st.rerun()

# Reset button to clear selections
if st.sidebar.button("Reset filters"):
    for k, v in DEFAULT_FILTERS.items():
        st.session_state[k] = v
    st.rerun()

brands = ["All"] + sorted(brand_summary["brand_clean"].unique().tolist())
selected_brand = st.sidebar.selectbox("Brand", brands, index=0, key="brand")

product_types = ["All"] + sorted(core["product_type"].unique().tolist())
selected_ptype = st.sidebar.selectbox("Product type", product_types, index=0, key="ptype")

price_tiers = ["All"] + [str(x) for x in price_tier_summary["price_tier"].dropna().astype(str).tolist()]
selected_ptier = st.sidebar.selectbox("Price tier", price_tiers, index=0, key="ptier")

only_auto_targeted = st.sidebar.checkbox("Only automotive-targeted", value=False, key="only_auto")
only_large_screen = st.sidebar.checkbox("Only large-screen-like", value=False, key="only_large")
only_rechargeable = st.sidebar.checkbox("Only rechargeable", value=False, key="only_recharge")

# ---------- APPLY FILTERS ----------
filtered = core.copy()

if selected_brand != "All":
    filtered = filtered[filtered["brand_clean"] == selected_brand]

if selected_ptype != "All":
    filtered = filtered[filtered["product_type"] == selected_ptype]

if selected_ptier != "All":
    filtered = filtered[filtered["price_tier"].astype(str) == selected_ptier]

if only_auto_targeted:
    filtered = filtered[filtered["is_automotive_targeted"]]

if only_large_screen:
    filtered = filtered[filtered["is_large_screen_like"]]

if only_rechargeable:
    filtered = filtered[filtered["is_rechargeable"]]

if not filtered.empty:
    f_total_revenue = filtered["ASIN Revenue"].sum()
    f_total_sales = filtered["ASIN Sales"].sum()
    f_asin_count = filtered["ASIN"].nunique()
    f_avg_price = filtered["Price"].mean()
    f_avg_rating = filtered["Reviews Rating"].mean() if "Reviews Rating" in filtered.columns else np.nan
else:
    f_total_revenue = f_total_sales = f_asin_count = 0
    f_avg_price = f_avg_rating = np.nan

# ---------- KPI CARDS ----------
st.subheader("Market KPIs – Current View")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total revenue (USD)", f"${f_total_revenue:,.0f}")
col2.metric("Units sold", f"{f_total_sales:,.0f}")
col3.metric("# of ASINs", f"{f_asin_count}")
col4.metric("Avg price", f"${f_avg_price:,.2f}" if not np.isnan(f_avg_price) else "N/A")
col5.metric("Avg rating", f"{f_avg_rating:.2f}" if not np.isnan(f_avg_rating) else "N/A")


st.markdown("---")
st.subheader("Display feature economics")

col_ls, col_cd, col_batt = st.columns(3)

with col_ls:
    st.caption("Large-screen vs non-large-screen: price and sales comparison (market-wide)")
    # 只显示关键几列
    st.dataframe(
        large_screen_summary[["segment", "asin_count", "total_sales", "total_revenue", "avg_price", "avg_rating"]],
        hide_index=True,
        use_container_width=True,
    )

with col_cd:
    st.caption("Color vs non-color display: price premium and sales comparison (market-wide)")
    st.dataframe(
        color_display_summary[["segment", "asin_count", "total_sales", "total_revenue", "avg_price", "avg_rating"]],
        hide_index=True,
        use_container_width=True,
    )

with col_batt:
    st.caption("Rechargeable vs non-rechargeable: price and sales comparison (market-wide)")
    st.dataframe(
        battery_category_summary[["segment", "asin_count", "total_sales", "total_revenue", "avg_price", "avg_rating"]],
        hide_index=True,
        use_container_width=True,
    )

# ---------- CHARTS ----------
st.markdown("---")
st.subheader("Brand performance")

brand_view = (
    filtered.groupby("brand_clean", as_index=False)
    .agg(total_revenue=("ASIN Revenue", "sum"), total_sales=("ASIN Sales", "sum"), asin_count=("ASIN", "nunique"))
    .sort_values("total_revenue", ascending=False)
    .head(15)
)

colA, colB = st.columns(2)
with colA:
    st.caption("Top brands by revenue (current filters)")
    st.bar_chart(brand_view.set_index("brand_clean")["total_revenue"], height=350)
with colB:
    st.caption("Top brands by units (current filters)")
    st.bar_chart(brand_view.set_index("brand_clean")["total_sales"], height=350)

colC_pie, colD_pie = st.columns(2)
with colC_pie:
    st.caption("Revenue share by brand")
    render_pie_chart(brand_view, "brand_clean", "total_revenue", "Revenue (USD)")
with colD_pie:
    st.caption("Units share by brand")
    render_pie_chart(brand_view, "brand_clean", "total_sales", "Units sold")

st.markdown("---")
st.subheader("Product type & price tiers")

pt_view = (
    filtered.groupby("product_type", as_index=False)
    .agg(total_revenue=("ASIN Revenue", "sum"), total_sales=("ASIN Sales", "sum"))
    .sort_values("total_revenue", ascending=False)
)

ptier_view = pd.DataFrame(columns=["price_tier", "total_revenue", "total_sales"])
if "price_tier" in filtered.columns:
    ptier_view = (
        filtered.dropna(subset=["price_tier"])
        .groupby("price_tier", as_index=False, observed=False)
        .agg(total_revenue=("ASIN Revenue", "sum"), total_sales=("ASIN Sales", "sum"))
        .sort_values("total_revenue", ascending=False)
    )
    ptier_view["price_tier"] = ptier_view["price_tier"].astype(str)

colC, colD = st.columns(2)
with colC:
    st.caption("Revenue share by product type")
    render_pie_chart(pt_view, "product_type", "total_revenue", "Revenue (USD)")
with colD:
    st.caption("Revenue share by price tier")
    render_pie_chart(ptier_view, "price_tier", "total_revenue", "Revenue (USD)")

colE, colF = st.columns(2)
with colE:
    st.caption("Units share by product type")
    render_pie_chart(pt_view, "product_type", "total_sales", "Units sold")
with colF:
    st.caption("Units share by price tier")
    render_pie_chart(ptier_view, "price_tier", "total_sales", "Units sold")

# ---------- TOP ASINS TABLE ----------
st.markdown("---")
st.subheader("Top ASINs (current filters)")

top_filtered = (
    filtered.sort_values("ASIN Revenue", ascending=False)
    .loc[
        :,
        [
            "ASIN",
            "Title",
            "brand_clean",
            "product_type",
            "Price",
            "ASIN Sales",
            "ASIN Revenue",
            "Review Count",
            "Reviews Rating",
            "is_large_screen_like",
            "is_rechargeable",
            "is_automotive_targeted",
        ],
    ]
    .head(50)
)

st.dataframe(top_filtered, width="stretch", hide_index=True)
st.caption("Showing up to 50 ASINs, sorted by revenue. Use the sidebar filters to focus on a specific segment or competitor.")
