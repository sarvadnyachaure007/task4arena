from pathlib import Path
import sys
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "sales_data.csv"
OUT = BASE / "visualizations"
REPORT = BASE / "report"

REQUIRED = [
    "order_id", "date", "product", "category",
    "city", "quantity", "unit_price"
]


def load_and_clean(path):
    """Load CSV, validate its structure, and clean invalid rows."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The input dataset is empty.")

    missing_columns = set(REQUIRED) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    original_count = len(df)
    df = df.drop_duplicates()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["quantity", "unit_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_columns = ["order_id", "product", "category", "city"]
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    df = df.dropna(subset=REQUIRED)
    df = df[
        (df["quantity"] > 0) &
        (df["unit_price"] >= 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid records remain after cleaning.")

    df["revenue"] = df["quantity"] * df["unit_price"]
    df["month"] = df["date"].dt.to_period("M").astype(str)

    print(f"Rows loaded: {original_count}")
    print(f"Valid rows: {len(df)}")
    print(f"Rows removed: {original_count - len(df)}")

    return df


def analyze(df):
    """Calculate category, monthly, and overall business metrics."""
    category_sales = (
        df.groupby("category", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )

    monthly_sales = (
        df.groupby("month", as_index=False)["revenue"]
        .sum()
        .sort_values("month")
    )

    total_revenue = df["revenue"].sum()
    total_units = df["quantity"].sum()
    average_order_value = df["revenue"].mean()

    return {
        "category_sales": category_sales,
        "monthly_sales": monthly_sales,
        "total_revenue": total_revenue,
        "total_units": total_units,
        "average_order_value": average_order_value,
        "orders": len(df),
    }


def visualize(results):
    """Create three professionally formatted charts."""
    OUT.mkdir(parents=True, exist_ok=True)

    category = results["category_sales"]
    monthly = results["monthly_sales"]

    plt.style.use("ggplot")

    # 1. Bar chart: revenue by category
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        category["category"], category["revenue"],
        color="#377eb8"
    )
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_title("Total Sales Revenue by Product Category")
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Revenue")
    fig.tight_layout()
    fig.savefig(OUT / "sales_by_category.png", dpi=160)
    plt.close(fig)

    # 2. Line chart: monthly revenue
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        monthly["month"], monthly["revenue"],
        marker="o", linewidth=2, color="#e68632"
    )
    ax.set_title("Monthly Sales Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(OUT / "monthly_sales_trend.png", dpi=160)
    plt.close(fig)

    # 3. Pie chart: category share of revenue
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        category["revenue"],
        labels=category["category"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Revenue Distribution by Category")
    fig.tight_layout()
    fig.savefig(OUT / "category_revenue_distribution.png", dpi=160)
    plt.close(fig)


def create_report(df, results):
    """Save cleaned data, summary CSVs, and written insights."""
    REPORT.mkdir(parents=True, exist_ok=True)

    df.to_csv(REPORT / "cleaned_sales_data.csv", index=False)
    results["category_sales"].to_csv(
        REPORT / "category_sales.csv", index=False
    )
    results["monthly_sales"].to_csv(
        REPORT / "monthly_sales.csv", index=False
    )

    category = results["category_sales"]
    monthly = results["monthly_sales"]

    top_category = category.iloc[0]
    best_month = monthly.loc[monthly["revenue"].idxmax()]
    lowest_month = monthly.loc[monthly["revenue"].idxmin()]

    share = (
        top_category["revenue"] / results["total_revenue"] * 100
        if results["total_revenue"] else 0
    )

    insights = [
        f"- **Top category:** {top_category['category']} "
        f"generated {top_category['revenue']:,.2f} in revenue.",
        f"- **Category contribution:** "
        f"{top_category['category']} accounted for {share:.1f}% "
        f"of total revenue.",
        f"- **Best month:** {best_month['month']} "
        f"with revenue of {best_month['revenue']:,.2f}.",
        f"- **Lowest month:** {lowest_month['month']} "
        f"with revenue of {lowest_month['revenue']:,.2f}.",
        "- **Business recommendation:** Compare category demand "
        "and monthly performance when planning inventory and promotions."
    ]

    report = f"""# E-commerce Sales Analysis Report

## 1. Executive Summary

This project analyzes a synthetic e-commerce dataset using Python,
Pandas, and Matplotlib. Revenue is calculated as quantity multiplied
by unit price.

## 2. Key Performance Indicators

| Metric | Value |
|---|---:|
| Valid sales records | {results['orders']:,} |
| Total units sold | {results['total_units']:,.0f} |
| Total revenue | {results['total_revenue']:,.2f} |
| Average revenue per record | {results['average_order_value']:,.2f} |

*Currency is unspecified in the sample data.*

## 3. Written Insights

{chr(10).join(insights)}

## 4. Visualizations

### Revenue by category
![Category sales](../visualizations/sales_by_category.png)

### Monthly revenue trend
![Monthly trend](../visualizations/monthly_sales_trend.png)

### Category revenue distribution
![Category distribution](../visualizations/category_revenue_distribution.png)

## 5. Data Cleaning

- Checked that required columns were present.
- Removed duplicate rows.
- Converted dates and numeric columns to suitable data types.
- Removed rows with missing required fields.
- Excluded non-positive quantities and negative unit prices.
- Created revenue and month fields.

## 6. Tools Used

- Python
- Pandas
- Matplotlib

## 7. Limitations

The dataset is synthetic and contains only a limited number of
sales records. Revenue does not account for discounts, taxes,
returns, shipping costs, or profit margins. Findings should not
be treated as conclusions about a real business.

## 8. Conclusion

The charts and summary tables help compare product categories
and identify monthly revenue patterns. Further analysis could
include profit, customer behavior, and larger real-world datasets.
"""

    (REPORT / "analysis_report.md").write_text(
        report, encoding="utf-8"
    )

    print("\nKEY FINDINGS")
    print(f"Total revenue: {results['total_revenue']:,.2f}")
    print(f"Units sold: {results['total_units']:,.0f}")
    print(f"Top category: {top_category['category']}")
    print(f"Best month: {best_month['month']}")
    print(f"\nReport saved to: {REPORT / 'analysis_report.md'}")


def main():
    try:
        for folder in [BASE / "data", OUT, REPORT]:
            folder.mkdir(parents=True, exist_ok=True)

        df = load_and_clean(DATA)
        results = analyze(df)
        visualize(results)
        create_report(df, results)

        print("\nProject completed successfully!")
        print(f"Charts: {OUT}")
        print(f"Report: {REPORT}")

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"Data error: {exc}", file=sys.stderr)
        sys.exit(1)
    except PermissionError as exc:
        print(f"Permission error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
