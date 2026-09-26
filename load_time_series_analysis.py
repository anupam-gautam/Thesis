from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "DataSet" / "RAW_smartgrid_5yr_hourly_natural.csv"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekday_name'] = df['timestamp'].dt.day_name()
    return df


def plot_a_entire_series(df):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df['timestamp'], df['load_kw'], color='lightgray', linewidth=0.8, alpha=0.8, label='Raw load')
    rolling = df['load_kw'].rolling(window=24 * 7, min_periods=1).mean()
    ax.plot(df['timestamp'], rolling, color='tab:red', linewidth=2, label='7-day rolling mean')
    ax.set_title('Plot A: Entire series of electricity load')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Load (kW)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'plot_a_entire_series.png', dpi=200)
    plt.close(fig)


def plot_b_one_week(df):
    start = df['timestamp'].min() + pd.Timedelta(days=10)
    end = start + pd.Timedelta(days=7)
    week_df = df[(df['timestamp'] >= start) & (df['timestamp'] < end)].copy()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(week_df['timestamp'], week_df['load_kw'], color='tab:blue', linewidth=1.8)

    for day in week_df['timestamp'].dt.normalize().drop_duplicates():
        mask = week_df['timestamp'].dt.normalize() == day
        if week_df.loc[mask, 'dayofweek'].iloc[0] >= 5:
            start_x = pd.Timestamp(day)
            end_x = start_x + pd.Timedelta(days=1)
            ax.axvspan(start_x, end_x, color='lightgray', alpha=0.25)

    ax.set_title('Plot B: One week view')
    ax.set_xlabel('Time')
    ax.set_ylabel('Load (kW)')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'plot_b_one_week.png', dpi=200)
    plt.close(fig)


def plot_c_one_month(df):
    month_start = df['timestamp'].min().to_period('M').to_timestamp()
    month_df = df[(df['timestamp'] >= month_start) & (df['timestamp'] < month_start + pd.DateOffset(months=1))].copy()
    monthly_daily = month_df.set_index('timestamp').resample('D')['load_kw'].mean()

    fig, ax = plt.subplots(figsize=(16, 5))
    monthly_daily.plot(ax=ax, color='tab:green', linewidth=2)
    ax.set_title('Plot C: One month daily average')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average daily load (kW)')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'plot_c_one_month.png', dpi=200)
    plt.close(fig)


def plot_d_daily_and_hourly_profiles(df):
    daily_avg = df.set_index('timestamp').resample('D')['load_kw'].mean()
    hourly_avg = df.groupby('hour')['load_kw'].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    daily_avg.plot(ax=axes[0], color='tab:purple', linewidth=2)
    axes[0].set_title('Plot D1: Daily average over time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Average load (kW)')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    axes[1].bar(hourly_avg.index, hourly_avg.values, color='tab:orange')
    axes[1].set_title('Plot D2: Average load by hour of day')
    axes[1].set_xlabel('Hour of day')
    axes[1].set_ylabel('Average load (kW)')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'plot_d_daily_and_hourly_profiles.png', dpi=200)
    plt.close(fig)


def plot_e_weekday_profile_heatmap(df):
    weekday_hour = df.pivot_table(index='hour', columns='dayofweek', values='load_kw', aggfunc='mean')
    weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekday_hour.columns = weekday_labels

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(weekday_hour.values, cmap='YlOrRd', aspect='auto')
    ax.set_title('Plot E: Weekday × hour heatmap')
    ax.set_xlabel('Weekday')
    ax.set_ylabel('Hour of day')
    ax.set_xticks(range(len(weekday_labels)))
    ax.set_xticklabels(weekday_labels)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([str(h) for h in range(0, 24, 2)])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Average load (kW)')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'plot_e_weekday_hour_heatmap.png', dpi=200)
    plt.close(fig)


def print_summary(df):
    first = df['load_kw'].iloc[0]
    last = df['load_kw'].iloc[-1]
    percent_change = ((last - first) / first) * 100

    daily_avg = df.set_index('timestamp').resample('D')['load_kw'].mean()
    hourly_avg = df.groupby('hour')['load_kw'].mean()
    peak_hour = hourly_avg.idxmax()
    peak_load = hourly_avg.max()
    weekend_avg = df[df['dayofweek'] >= 5]['load_kw'].mean()
    weekday_avg = df[df['dayofweek'] < 5]['load_kw'].mean()

    print('=== Dataset summary ===')
    print(f'Shape: {df.shape}')
    print(f'Time range: {df["timestamp"].min()} to {df["timestamp"].max()}')
    print(f'Average load: {df["load_kw"].mean():.2f} kW')
    print(f'Min load: {df["load_kw"].min():.2f} kW')
    print(f'Max load: {df["load_kw"].max():.2f} kW')
    print(f'Overall trend change from first to last value: {percent_change:.2f}%')
    print(f'Peak average hour of day: hour {peak_hour} with {peak_load:.2f} kW')
    print(f'Average weekday load: {weekday_avg:.2f} kW')
    print(f'Average weekend load: {weekend_avg:.2f} kW')
    print(f'Daily average trend min/max: {daily_avg.min():.2f} kW -> {daily_avg.max():.2f} kW')
    print('\nPlots saved in:', PLOTS_DIR)


def main():
    df = load_data()
    plot_a_entire_series(df)
    plot_b_one_week(df)
    plot_c_one_month(df)
    plot_d_daily_and_hourly_profiles(df)
    plot_e_weekday_profile_heatmap(df)
    print_summary(df)


if __name__ == '__main__':
    main()
