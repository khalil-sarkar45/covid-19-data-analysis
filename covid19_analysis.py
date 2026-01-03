import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# =============================
# FILE PATHS
# =============================
CONFIRMED_PATH = "dataset/time_series_covid19_confirmed_global.csv"
DEATHS_PATH = "dataset/time_series_covid19_deaths_global.csv"


# =============================
# DATA ACQUISITION
# =============================
def load_data():
    confirmed = pd.read_csv(CONFIRMED_PATH)
    deaths = pd.read_csv(DEATHS_PATH)
    return confirmed, deaths


def run_analysis_pipeline(confirmed, deaths):

    # =============================
    # PHASE 1: DATA ACQUISITION & INSPECTION
    # =============================
    print("\n--- PHASE 1: DATA ACQUISITION & INSPECTION ---")
    print("Confirmed Data Sample:")
    print(confirmed.head())
    print("\nDeaths Data Info:")
    print(deaths.info())

    # =============================
    # PHASE 2: DATA CLEANING
    # =============================
    print("\n--- PHASE 2: DATA CLEANING ---")

    confirmed.fillna(0, inplace=True)
    deaths.fillna(0, inplace=True)

    confirmed.drop(columns=["Lat", "Long"], inplace=True)
    deaths.drop(columns=["Lat", "Long"], inplace=True)

    print("Cleaning completed.")
    print("Confirmed shape:", confirmed.shape)
    print("Deaths shape:", deaths.shape)

    # =============================
    # PHASE 3: DATA TRANSFORMATION
    # =============================
    print("\n--- PHASE 3: DATA TRANSFORMATION ---")

    confirmed_grouped = confirmed.groupby("Country/Region").sum(numeric_only=True)
    deaths_grouped = deaths.groupby("Country/Region").sum(numeric_only=True)

    confirmed_ts = confirmed_grouped.T
    deaths_ts = deaths_grouped.T

    confirmed_ts.index = pd.to_datetime(confirmed_ts.index)
    deaths_ts.index = pd.to_datetime(deaths_ts.index)

    print("Transformation completed successfully.")

    # =============================
    # PHASE 4: DATA ANALYSIS
    # =============================
    print("\n--- PHASE 4: DATA ANALYSIS ---")

    total_confirmed = confirmed_ts.iloc[-1].sort_values(ascending=False)
    total_deaths = deaths_ts.iloc[-1].sort_values(ascending=False)

    print("\nTop 5 Countries by Confirmed Cases:")
    print(total_confirmed.head())

    print("\nTop 5 Countries by Deaths:")
    print(total_deaths.head())

    # =============================
    # PHASE 5: DATA COMPARISON
    # =============================
    print("\n--- PHASE 5: DATA COMPARISON ---")

    comparison_df = pd.DataFrame({
        "Confirmed": total_confirmed,
        "Deaths": total_deaths
    })

    comparison_df["Death Rate (%)"] = (
        comparison_df["Deaths"] / comparison_df["Confirmed"]
    ) * 100

    print("\nTop 5 Countries by Death Rate:")
    print(comparison_df.sort_values("Death Rate (%)", ascending=False).head())

    # =============================
    # PHASE 6: DATA VISUALIZATION
    # =============================
    print("\n--- PHASE 6: DATA VISUALIZATION ---")

    sns.set_style("whitegrid")

    # Plot 1: Global Trend
    plt.figure(figsize=(14, 6))
    confirmed_ts.sum(axis=1).plot()
    plt.title("Global COVID-19 Confirmed Cases Over Time")
    plt.xlabel("Date")
    plt.ylabel("Total Cases")
    plt.show()

    # Plot 2: Top 10 Countries
    plt.figure(figsize=(10, 5))
    total_confirmed.head(10).plot(kind="bar")
    plt.title("Top 10 Countries by Confirmed COVID-19 Cases")
    plt.xlabel("Country")
    plt.ylabel("Cases")
    plt.show()

    # Plot 3: Death Rate Comparison
    plt.figure(figsize=(10, 5))
    comparison_df.sort_values("Death Rate (%)", ascending=False).head(10)["Death Rate (%)"]\
        .plot(kind="bar")
    plt.title("Top 10 Countries by COVID-19 Death Rate")
    plt.ylabel("Death Rate (%)")
    plt.show()

    # Plot 4: Heatmap (Top 10 Countries)
    plt.figure(figsize=(12, 6))
    top10_countries = confirmed_ts[total_confirmed.head(10).index]
    sns.heatmap(top10_countries.T, cmap="Reds")
    plt.title("COVID-19 Spread Heatmap (Top 10 Countries)")
    plt.xlabel("Date")
    plt.ylabel("Country")
    plt.show()


# =============================
# MAIN EXECUTION
# =============================
if __name__ == "__main__":
    confirmed_data, deaths_data = load_data()
    run_analysis_pipeline(confirmed_data, deaths_data)
