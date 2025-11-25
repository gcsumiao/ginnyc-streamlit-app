import pandas as pd
import streamlit as st
from pathlib import Path

# ======== PATHS & SETTINGS ========
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw_data"
SUBCATEGORY_DEFAULT = "Thermal Imagers"
TOP_N = 50
# ==================================


@st.cache_data
def load_data(raw_dir: Path) -> pd.DataFrame:
    """Load and combine all CSV files in raw_data folder."""
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns once to keep filters reliable.
    numeric_cols = [
        "Price",
        "ASIN Revenue",
        "ASIN Sales",
        "Review Count",
        "Reviews Rating",
        "Sales Trend (90 days) (%)",
        "Price Trend (90 days) (%)",
        "Age (Month)",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "Brand" in data.columns:
        data["Brand"] = data["Brand"].fillna("Unknown")

    return data


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    reset_clicked = st.sidebar.button("Reset all filters")
    if reset_clicked:
        st.session_state.clear()
        st.experimental_rerun()

    view_all = st.sidebar.checkbox("View all data", value=st.session_state.get("view_all", False), key="view_all")
    if view_all:
        return df.copy()

    if "Subcategory" in df.columns:
        sub_opts = ["All"] + sorted(df["Subcategory"].dropna().unique().tolist())
        default_idx = sub_opts.index(SUBCATEGORY_DEFAULT) if SUBCATEGORY_DEFAULT in sub_opts else 0
        sub_choice = st.sidebar.selectbox("Subcategory", sub_opts, index=default_idx)
        if sub_choice != "All":
            filtered = filtered[filtered["Subcategory"] == sub_choice]

    if "Brand" in df.columns:
        brand_opts = sorted(filtered["Brand"].dropna().unique().tolist())
        brand_opts_with_all = ["All Brands"] + brand_opts
        brand_choice = st.sidebar.multiselect("Brand", brand_opts_with_all, default=["All Brands"], key="brand_filter")
        if brand_choice and "All Brands" not in brand_choice:
            filtered = filtered[filtered["Brand"].isin(brand_choice)]

    if "Price" in df.columns and not filtered["Price"].isna().all():
        price_series = filtered["Price"].dropna()
        if not price_series.empty:
            min_price = float(price_series.min())
            max_price = float(price_series.max())
            if min_price == max_price:
                st.sidebar.info(f"Price fixed at ${min_price:,.2f} for the current filters.")
            else:
                price_range = st.sidebar.slider(
                    "Price range ($)",
                    min_value=min_price,
                    max_value=max_price,
                    value=(min_price, max_price),
                    format="$%0.2f",
                )
                filtered = filtered[(filtered["Price"] >= price_range[0]) & (filtered["Price"] <= price_range[1])]

    if "Reviews Rating" in df.columns and not filtered["Reviews Rating"].isna().all():
        rating_range = st.sidebar.slider("Rating", 0.0, 5.0, (0.0, 5.0), step=0.1)
        filtered = filtered[
            (filtered["Reviews Rating"] >= rating_range[0]) & (filtered["Reviews Rating"] <= rating_range[1])
        ]

    if "Fulfillment" in df.columns:
        fulfill_opts = sorted(filtered["Fulfillment"].dropna().unique().tolist())
        fulfill_choice = st.sidebar.multiselect("Fulfillment", fulfill_opts, default=fulfill_opts)
        if fulfill_choice:
            filtered = filtered[filtered["Fulfillment"].isin(fulfill_choice)]

    return filtered


def kpi_block(df: pd.DataFrame):
    if df.empty:
        st.warning("No data after filters.")
        return

    total_units = df["ASIN Sales"].sum() if "ASIN Sales" in df.columns else 0
    total_revenue = df["ASIN Revenue"].sum() if "ASIN Revenue" in df.columns else 0
    revenue_last_30 = total_revenue  # dataset is monthly, so we treat ASIN Revenue as last 30 days
    brand_count = df["Brand"].nunique(dropna=True) if "Brand" in df.columns else 0
    asin_count = df["ASIN"].nunique(dropna=True) if "ASIN" in df.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Units Sold", f"{int(total_units):,}")
    col2.metric("Total Revenue (est.)", f"${total_revenue:,.0f}")
    col3.metric("Revenue (last 30 days)", f"${revenue_last_30:,.0f}")
    col4.metric("Total Brands", f"{brand_count:,}")
    col5.metric("Total ASINs", f"{asin_count:,}")

    st.caption("Revenue fields are interpreted as monthly/last 30 days in the raw export.")


def build_top_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if metric_col not in df.columns or df.empty:
        return pd.DataFrame()

    sorted_df = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    top = sorted_df.head(TOP_N).copy()

    keep_cols = [
        "ASIN",
        "Title",
        "Brand",
        "Price",
        "ASIN Revenue",
        "ASIN Sales",
        "Review Count",
        "Reviews Rating",
        "URL",
    ]
    keep_cols = [c for c in keep_cols if c in top.columns]
    top = top[keep_cols]
    top.insert(0, "Ranking", range(1, len(top) + 1))
    return top


def charts(df: pd.DataFrame):
    import altair as alt

    if df.empty:
        st.info("No chart data for the current filters.")
        return

    if "ASIN Revenue" in df.columns and "Brand" in df.columns:
        brand_rev = (
            df.groupby("Brand")["ASIN Revenue"]
            .sum()
            .reset_index()
            .sort_values("ASIN Revenue", ascending=False)
            .head(10)
        )
        bar = (
            alt.Chart(brand_rev)
            .mark_bar(color="#4E79A7")
            .encode(x=alt.X("ASIN Revenue:Q", title="Est. Monthly Revenue"), y=alt.Y("Brand:N", sort="-x"))
            .properties(height=300, title="Top 10 Brands by Revenue")
        )
        st.altair_chart(bar, use_container_width=True)

    def brand_share_pie(metric: str, title: str, value_title: str):
        if metric not in df.columns:
            return None
        share = (
            df.groupby("Brand")[metric]
            .sum()
            .reset_index()
            .sort_values(metric, ascending=False)
        )
        share = share[share[metric] > 0]
        if share.empty:
            return None
        if len(share) > 8:
            top = share.head(8)
            other_total = share[metric].iloc[8:].sum()
            if other_total > 0:
                share = pd.concat([top, pd.DataFrame({"Brand": ["Other"], metric: [other_total]})], ignore_index=True)
            else:
                share = top
        share["Share (%)"] = (share[metric] / share[metric].sum()) * 100
        return (
            alt.Chart(share)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta(f"{metric}:Q", title=value_title),
                color=alt.Color("Brand:N", legend=alt.Legend(title="Brand")),
                tooltip=[
                    "Brand",
                    alt.Tooltip(f"{metric}:Q", title=value_title, format=",.0f"),
                    alt.Tooltip("Share (%):Q", title="Share (%)", format=".1f"),
                ],
            )
            .properties(title=title, height=280)
        )

    pie_rev = brand_share_pie("ASIN Revenue", "Monthly Revenue Share by Brand", "Est. Revenue")
    pie_units = brand_share_pie("ASIN Sales", "Monthly Units Share by Brand", "Units Sold")
    if pie_rev or pie_units:
        col1, col2 = st.columns(2)
        if pie_rev:
            col1.altair_chart(pie_rev, use_container_width=True)
        if pie_units:
            col2.altair_chart(pie_units, use_container_width=True)

    if "Price" in df.columns and df["Price"].notna().any():
        price_df = df.dropna(subset=["Price"]).copy()
        prices = price_df["Price"]
        if prices.nunique() > 1:
            bins = min(6, prices.nunique())
            try:
                tiers = pd.qcut(prices, q=bins, duplicates="drop")
            except ValueError:
                tiers = None
            if tiers is not None:
                tier_labels = [f"${iv.left:,.0f} - ${iv.right:,.0f}" for iv in tiers.cat.categories]
                tier_map = dict(zip(tiers.cat.categories, tier_labels))
                price_df["Price Tier"] = tiers.map(tier_map)
                price_df["Price Tier"] = pd.Categorical(price_df["Price Tier"], categories=tier_labels, ordered=True)

                agg_config = {}
                if "ASIN Sales" in price_df.columns:
                    agg_config["Units Sold"] = ("ASIN Sales", "sum")
                if "ASIN Revenue" in price_df.columns:
                    agg_config["Revenue"] = ("ASIN Revenue", "sum")
                if "Reviews Rating" in price_df.columns:
                    agg_config["Avg Rating"] = ("Reviews Rating", "mean")

                if agg_config:
                    agg = price_df.groupby("Price Tier").agg(**agg_config).reset_index()

                    if "Units Sold" in agg.columns:
                        bar_tooltips = ["Price Tier"]
                        if "Units Sold" in agg.columns:
                            bar_tooltips.append(alt.Tooltip("Units Sold:Q", format=",.0f"))
                        if "Revenue" in agg.columns:
                            bar_tooltips.append(alt.Tooltip("Revenue:Q", format=",.0f"))
                        if "Avg Rating" in agg.columns:
                            bar_tooltips.append(alt.Tooltip("Avg Rating:Q", format=".2f"))
                        bars = (
                            alt.Chart(agg)
                            .mark_bar(color="#4E79A7")
                            .encode(
                                x=alt.X("Price Tier:N", sort=tier_labels, title="Price tier"),
                                y=alt.Y("Units Sold:Q", title="Units sold"),
                                tooltip=bar_tooltips,
                            )
                            .properties(height=300, title="Units sold by price tier")
                        )
                    else:
                        bars = None

                    if "Avg Rating" in agg.columns:
                        rating_line = (
                            alt.Chart(agg)
                            .mark_line(color="#F28E2B", point=True)
                            .encode(
                                x=alt.X("Price Tier:N", sort=tier_labels, title="Price tier"),
                                y=alt.Y("Avg Rating:Q", title="Avg rating", scale=alt.Scale(domain=[0, 5])),
                                tooltip=["Price Tier", alt.Tooltip("Avg Rating:Q", format=".2f")],
                            )
                        )
                    else:
                        rating_line = None

                    if bars and rating_line:
                        combo = alt.layer(bars, rating_line).resolve_scale(y="independent")
                        st.altair_chart(combo, use_container_width=True)
                    elif bars:
                        st.altair_chart(bars, use_container_width=True)

                if "Reviews Rating" in price_df.columns:
                    rating_box = (
                        alt.Chart(price_df.dropna(subset=["Reviews Rating"]))
                        .mark_boxplot(color="#59A14F")
                        .encode(
                            x=alt.X("Price Tier:N", sort=tier_labels, title="Price tier"),
                            y=alt.Y("Reviews Rating:Q", title="Rating"),
                            tooltip=["Price Tier", "Reviews Rating"],
                        )
                        .properties(height=300, title="Rating distribution by price tier")
                    )
                    st.altair_chart(rating_box, use_container_width=True)


def main():
    st.set_page_config(page_title="Thermal Imager Dashboard", layout="wide")
    st.title("Thermal Imager Performance Dashboard")
    st.caption("Interactive view of revenue and units. Adjust filters in the sidebar.")

    df = load_data(RAW_DIR)
    if df.empty:
        st.error(f"No CSV files found in {RAW_DIR}")
        return

    with st.sidebar:
        st.header("Filters")
    filtered = apply_filters(df)

    kpi_block(filtered)

    charts(filtered)

    st.subheader(f"Top {TOP_N} Rank by Monthly Revenue")
    top_rev = build_top_table(filtered, "ASIN Revenue")
    if not top_rev.empty:
        st.dataframe(top_rev, use_container_width=True)
    else:
        st.info("No revenue data available for the current filter set.")

    st.subheader(f"Top {TOP_N} Rank by Monthly Units")
    top_units = build_top_table(filtered, "ASIN Sales")
    if not top_units.empty:
        st.dataframe(top_units, use_container_width=True)
    else:
        st.info("No unit sales data available for the current filter set.")


if __name__ == "__main__":
    main()
