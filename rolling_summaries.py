import os
import pandas as pd

BASE_DIR = r"correlation_outputs"
THRESHOLD = 0.7
MIN_DURATION = 5

def process_directory(rolling_dir):
    summary_data = []

    for filename in os.listdir(rolling_dir):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(rolling_dir, filename)
        df = pd.read_csv(filepath, parse_dates=["Date"])
        series = df['rolling_correlation'].dropna()

        if series.empty:
            continue

        pair_name = filename.replace("_rolling.csv", "")
        mean = series.mean()
        median = series.median()
        std = series.std()
        min_val = series.min()
        max_val = series.max()
        pct_above_0_7 = (series > THRESHOLD).mean() * 100

        # Add summary row
        base_row = {
            "pair": pair_name,
            "mean": round(mean, 4),
            "median": round(median, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "std": round(std, 4),
            "% > 0.7": round(pct_above_0_7, 2)
        }

        # Detect strong correlation periods
        df["is_strong"] = df["rolling_correlation"] > THRESHOLD
        streak_id = (df["is_strong"] != df["is_strong"].shift()).cumsum()
        streaks = df[df["is_strong"]].groupby(streak_id)

        for _, group in streaks:
            duration = len(group)
            if duration >= MIN_DURATION:
                start_date = group["Date"].iloc[0]
                end_date = group["Date"].iloc[-1]
                avg_corr = group["rolling_correlation"].mean()
                row = base_row.copy()
                row.update({
                    "strong_corr_start": start_date.strftime("%Y-%m-%d"),
                    "strong_corr_end": end_date.strftime("%Y-%m-%d"),
                    "strong_corr_duration": duration,
                    "strong_corr_avg": round(avg_corr, 4),
                    "month": start_date.strftime("%b"),
                    "year": start_date.year
                })
                summary_data.append(row)

        # If no strong period found, still include basic summary
        if not any(df["is_strong"]):
            summary_data.append(base_row)

    # Save merged summary
    summary_df = pd.DataFrame(summary_data)
    dir_name = os.path.basename(rolling_dir.rstrip(os.sep))
    output_file = os.path.join(BASE_DIR, f"{dir_name}_combined_summary.csv")
    summary_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

def process_all_rolling_dirs():
    for subdir in os.listdir(BASE_DIR):
        full_path = os.path.join(BASE_DIR, subdir)
        if os.path.isdir(full_path) and subdir.startswith("rolling_correlation"):
            process_directory(full_path)

if __name__ == "__main__":
    process_all_rolling_dirs()
