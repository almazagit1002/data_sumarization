import os
import pandas as pd
from scipy.stats import linregress

DATA_DIR = "C:\\Users\\Maza\\Desktop\\USD-strenght-analysis\\data"
SUMMARY_DIR = "summaries"
os.makedirs(SUMMARY_DIR, exist_ok=True)


def max_drawdown(series):
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max
    return drawdown.min()


def trend_slope(series):
    y = series.dropna().values
    x = range(len(y))
    if len(y) < 2:
        return 0
    slope, _, _, _, _ = linregress(x, y)
    return slope


def summarize_dataset(file_path, dataset_name):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    summary_lines = [f"SUMMARY REPORT FOR: {dataset_name.upper()}", "=" * 60]
    summary_lines.append(f"Date Range: {df.index.min().date()} to {df.index.max().date()}")
    summary_lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    for col in df.columns:
        series = df[col].dropna()

        if len(series) < 100:
            continue  # skip short series

        summary_lines.append(f"\n--- {col} ---")
        summary_lines.append(f"Latest Value: {series.iloc[-1]:.2f}")
        summary_lines.append(f"Value Range: {series.min():.2f} to {series.max():.2f}")
        summary_lines.append(f"MA20: {series.rolling(20).mean().iloc[-1]:.2f}")
        summary_lines.append(f"MA50: {series.rolling(50).mean().iloc[-1]:.2f}")
        summary_lines.append(f"MA200: {series.rolling(200).mean().iloc[-1]:.2f}")

        # Quarterly and yearly returns
        quarterly = series.resample("QE").last().pct_change().dropna()
        yearly = series.resample("YE").last().pct_change().dropna()
        summary_lines.append(f"QoQ Avg Return: {quarterly.mean() * 100:.2f}%")
        summary_lines.append(f"YoY Avg Return: {yearly.mean() * 100:.2f}%")

        # Volatility
        vol = series.pct_change().rolling(30).std().mean()
        summary_lines.append(f"30-Day Volatility: {vol * 100:.2f}%")

        # Drawdown
        drawdown = max_drawdown(series)
        summary_lines.append(f"Max Drawdown: {drawdown * 100:.2f}%")

        # Trend slope
        slope = trend_slope(series)
        summary_lines.append(f"Trend Slope (lin. reg.): {slope:.4f}")

    # Save to txt
    output_path = os.path.join(SUMMARY_DIR, f"{dataset_name.lower()}_summary.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"✅ Saved summary to {output_path}")


def main():
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            dataset_name = file.replace(".csv", "")
            file_path = os.path.join(DATA_DIR, file)
            summarize_dataset(file_path, dataset_name)


if __name__ == "__main__":
    main()
