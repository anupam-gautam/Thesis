"""Exploratory Data Analysis (EDA) plotting module for training data."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import seaborn as sns


class EDAPlotter:
    """Generate exploratory plots and diagnostics from training data."""
    
    def __init__(
        self,
        train_df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        target_col: str = 'load_kw',
        output_dir: str = 'outputs/plots/eda_plots'
    ):
        """
        Initialize EDA plotter.
        
        Args:
            train_df: Training data DataFrame (training set only)
            timestamp_col: Name of timestamp column
            target_col: Name of target (load) column
            output_dir: Output directory for plots
        """
        self.train_df = train_df.copy()
        self.train_df[timestamp_col] = pd.to_datetime(self.train_df[timestamp_col])
        self.train_df = self.train_df.sort_values(timestamp_col)
        
        self.timestamp_col = timestamp_col
        self.target_col = target_col
        self.output_dir = output_dir
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Cache diagnostics
        self._diagnostics: Dict[str, Any] = {}
    
    def generate_all_plots(self) -> Dict[str, str]:
        """
        Generate all EDA plots (A-E).
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}
        
        plot_paths['plot_a'] = self.plot_a_full_series_with_rolling_mean()
        plot_paths['plot_b'] = self.plot_b_complete_week()
        plot_paths['plot_c'] = self.plot_c_complete_month()
        plot_paths['plot_d1'] = self.plot_d1_daily_average()
        plot_paths['plot_d2'] = self.plot_d2_hourly_average()
        plot_paths['plot_e'] = self.plot_e_weekday_hour_heatmap()
        
        return plot_paths
    
    def plot_a_full_series_with_rolling_mean(self) -> str:
        """
        Plot A: Full series + rolling mean.
        
        Returns:
            Path to saved PNG file
        """
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Original load values (thin, light)
        ax.plot(
            self.train_df[self.timestamp_col],
            self.train_df[self.target_col],
            linewidth=0.8,
            alpha=0.5,
            label='Original Load',
            color='lightblue'
        )
        
        # 30-day rolling mean (thick, dark)
        rolling_mean = self.train_df[self.target_col].rolling(
            window=24*30, center=True
        ).mean()
        ax.plot(
            self.train_df[self.timestamp_col],
            rolling_mean,
            linewidth=2.5,
            label='30-day Rolling Mean',
            color='darkblue'
        )
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Load (kW)', fontsize=12, fontweight='bold')
        ax.set_title('Smart Grid Load 2020–2023 with 30-day Rolling Mean', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save PNG and SVG
        png_path = os.path.join(self.output_dir, 'plot_a_full_series.png')
        svg_path = os.path.join(self.output_dir, 'plot_a_full_series.svg')
        pdf_path = os.path.join(self.output_dir, 'plot_a_full_series.pdf')
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        # Export plot data
        self._export_plot_data(
            'plot_a',
            pd.DataFrame({
                'date': self.train_df[self.timestamp_col],
                'original_load': self.train_df[self.target_col],
                'rolling_mean_30d': rolling_mean
            })
        )
        
        return png_path
    
    def plot_b_complete_week(self) -> str:
        """
        Plot B: Complete week sample.
        
        Selects first complete week of training data deterministically.
        
        Returns:
            Path to saved PNG file
        """
        # Find first complete week (7 consecutive calendar days)
        train_df = self.train_df.sort_values(self.timestamp_col).reset_index(drop=True)
        train_df['date'] = train_df[self.timestamp_col].dt.date
        train_df['weekday'] = train_df[self.timestamp_col].dt.day_name()
        
        # Find first Monday
        first_monday_idx = train_df[train_df[self.timestamp_col].dt.weekday == 0].index[0]
        week_start_idx = first_monday_idx
        week_end_idx = week_start_idx + 7*24  # 7 days * 24 hours
        
        week_data = train_df.iloc[week_start_idx:week_end_idx].copy()
        week_data['hour_of_week'] = np.arange(len(week_data))
        week_data['day_name'] = week_data[self.timestamp_col].dt.day_name()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot with day separators
        for day_idx in range(7):
            day_start = day_idx * 24
            day_end = (day_idx + 1) * 24
            day_data = week_data.iloc[day_start:day_end]
            
            ax.plot(
                day_data['hour_of_week'],
                day_data[self.target_col],
                linewidth=2,
                marker='o',
                markersize=4,
                label=day_data['day_name'].iloc[0]
            )
            
            # Add vertical line at day boundaries
            if day_idx < 6:
                ax.axvline(x=day_end, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Hour within Week (0-167)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Load (kW)', fontsize=12, fontweight='bold')
        ax.set_title('Sample Week: Hourly Load Pattern', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save PNG and SVG
        png_path = os.path.join(self.output_dir, 'plot_b_week_sample.png')
        svg_path = os.path.join(self.output_dir, 'plot_b_week_sample.svg')
        pdf_path = os.path.join(self.output_dir, 'plot_b_week_sample.pdf')
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        # Export plot data
        export_data = week_data[[self.timestamp_col, self.target_col, 'hour_of_week', 'day_name']].copy()
        export_data.columns = ['timestamp', 'load_kw', 'hour_of_week', 'day_name']
        self._export_plot_data('plot_b', export_data)
        
        return png_path
    
    def plot_c_complete_month(self) -> str:
        """
        Plot C: Complete calendar month sample.
        
        Selects first complete calendar month (January 2020).
        
        Returns:
            Path to saved PNG file
        """
        # Get January 2020
        january = self.train_df[
            (self.train_df[self.timestamp_col].dt.month == 1) &
            (self.train_df[self.timestamp_col].dt.year == 2020)
        ].copy()
        
        january['day_of_month'] = january[self.timestamp_col].dt.day
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot daily patterns
        for day in range(1, 32):
            day_data = january[january['day_of_month'] == day]
            if len(day_data) > 0:
                ax.plot(
                    day_data[self.timestamp_col].dt.hour,
                    day_data[self.target_col],
                    linewidth=1.5,
                    alpha=0.6,
                    label=f'Jan {day}'
                )
        
        ax.set_xlabel('Hour of Day (0-23)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Load (kW)', fontsize=12, fontweight='bold')
        ax.set_title('Sample Month: Daily Load Pattern (January 2020)', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=4, loc='upper right')
        
        plt.tight_layout()
        
        # Save PNG and SVG
        png_path = os.path.join(self.output_dir, 'plot_c_month_sample.png')
        svg_path = os.path.join(self.output_dir, 'plot_c_month_sample.svg')
        pdf_path = os.path.join(self.output_dir, 'plot_c_month_sample.pdf')
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        # Export plot data
        export_data = january[[self.timestamp_col, self.target_col, 'day_of_month']].copy()
        export_data['hour_of_day'] = export_data[self.timestamp_col].dt.hour
        export_data = export_data[['day_of_month', 'hour_of_day', self.target_col]]
        export_data.columns = ['day_of_month', 'hour_of_day', 'load_kw']
        self._export_plot_data('plot_c', export_data)
        
        return png_path
    
    def plot_d1_daily_average(self) -> str:
        """
        Plot D1: Daily average load over time.
        
        Returns:
            Path to saved PNG file
        """
        daily_data = self.train_df.copy()
        daily_data['date'] = daily_data[self.timestamp_col].dt.date
        
        daily_avg = daily_data.groupby('date')[self.target_col].mean().reset_index()
        daily_avg['date'] = pd.to_datetime(daily_avg['date'])
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.plot(
            daily_avg['date'],
            daily_avg[self.target_col],
            linewidth=1.5,
            color='darkgreen',
            label='Daily Average Load'
        )
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Daily Average Load (kW)', fontsize=12, fontweight='bold')
        ax.set_title('Daily Average Load (2020–2023)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save PNG and SVG
        png_path = os.path.join(self.output_dir, 'plot_d1_daily_average.png')
        svg_path = os.path.join(self.output_dir, 'plot_d1_daily_average.svg')
        pdf_path = os.path.join(self.output_dir, 'plot_d1_daily_average.pdf')
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        # Export plot data
        self._export_plot_data('plot_d1', daily_avg.rename(columns={self.target_col: 'daily_avg_load'}))
        
        return png_path
    
    def plot_d2_hourly_average(self) -> str:
        """
        Plot D2: Hourly average + variability (±1 std).
        
        Returns:
            Path to saved PNG file
        """
        hourly_data = self.train_df.copy()
        hourly_data['hour'] = hourly_data[self.timestamp_col].dt.hour
        
        hourly_stats = hourly_data.groupby('hour')[self.target_col].agg(
            ['mean', 'std']
        ).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean
        ax.plot(
            hourly_stats['hour'],
            hourly_stats['mean'],
            linewidth=2.5,
            marker='o',
            markersize=8,
            color='darkred',
            label='Mean Load',
            zorder=3
        )
        
        # Add confidence band (±1 std)
        ax.fill_between(
            hourly_stats['hour'],
            hourly_stats['mean'] - hourly_stats['std'],
            hourly_stats['mean'] + hourly_stats['std'],
            alpha=0.3,
            color='red',
            label='±1 Std Dev'
        )
        
        ax.set_xlabel('Hour of Day (0-23)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Load (kW)', fontsize=12, fontweight='bold')
        ax.set_title('Average Load by Hour with ±1 Std Dev (Training Data)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save PNG and SVG
        png_path = os.path.join(self.output_dir, 'plot_d2_hourly_average.png')
        svg_path = os.path.join(self.output_dir, 'plot_d2_hourly_average.svg')
        pdf_path = os.path.join(self.output_dir, 'plot_d2_hourly_average.pdf')
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        # Export plot data
        self._export_plot_data('plot_d2', hourly_stats)
        
        return png_path
    
    def plot_e_weekday_hour_heatmap(self) -> str:
        """
        Plot E: Weekday × Hour heatmap.
        
        Returns:
            Path to saved PNG file
        """
        heatmap_data = self.train_df.copy()
        heatmap_data['weekday'] = heatmap_data[self.timestamp_col].dt.day_name()
        heatmap_data['hour'] = heatmap_data[self.timestamp_col].dt.hour
        
        # Create pivot table
        pivot = heatmap_data.pivot_table(
            values=self.target_col,
            index='weekday',
            columns='hour',
            aggfunc='mean'
        )
        
        # Reorder weekdays
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex(weekday_order)
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Create heatmap with different colormap for weekends
        im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(24))
        ax.set_yticks(range(7))
        ax.set_xticklabels(range(24))
        ax.set_yticklabels(weekday_order)
        
        # Highlight weekends with a border
        for i in [5, 6]:  # Saturday and Sunday
            rect = plt.Rectangle((-.5, i-.5), 24, 1, fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
        
        ax.set_xlabel('Hour of Day (0-23)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Weekday', fontsize=12, fontweight='bold')
        ax.set_title('Mean Load by Weekday & Hour', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Load (kW)')
        
        plt.tight_layout()
        
        # Save PNG and SVG
        png_path = os.path.join(self.output_dir, 'plot_e_heatmap.png')
        svg_path = os.path.join(self.output_dir, 'plot_e_heatmap.svg')
        pdf_path = os.path.join(self.output_dir, 'plot_e_heatmap.pdf')
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        
        # Export plot data
        self._export_plot_data('plot_e', pivot.reset_index().rename(columns={'index': 'weekday'}))
        
        return png_path
    
    def compute_diagnostics(self) -> Dict[str, Any]:
        """
        Compute diagnostic statistics from training data.
        
        Returns:
            Dictionary with diagnostic findings
        """
        diag = {}
        
        # Rolling mean and std (7-day and 30-day windows)
        load_series = self.train_df[self.target_col]
        diag['rolling_mean_7day'] = {
            'mean': load_series.rolling(window=7*24).mean().mean(),
            'std': load_series.rolling(window=7*24).mean().std()
        }
        diag['rolling_mean_30day'] = {
            'mean': load_series.rolling(window=30*24).mean().mean(),
            'std': load_series.rolling(window=30*24).mean().std()
        }
        
        # Autocorrelation at key lags
        diag['autocorrelation'] = {
            '24_hour_lag': float(load_series.autocorr(lag=24)),
            '168_hour_lag': float(load_series.autocorr(lag=168))
        }
        
        # Seasonal profiles
        hourly_data = self.train_df.copy()
        hourly_data['hour'] = hourly_data[self.timestamp_col].dt.hour
        hourly_stats = hourly_data.groupby('hour')[self.target_col].agg(['mean', 'std'])
        
        peak_hour = hourly_stats['mean'].idxmax()
        peak_load = hourly_stats.loc[peak_hour, 'mean']
        off_peak_hour = hourly_stats['mean'].idxmin()
        off_peak_load = hourly_stats.loc[off_peak_hour, 'mean']
        
        diag['seasonal_profiles'] = {
            'peak_hour': int(peak_hour),
            'peak_load_kw': float(peak_load),
            'off_peak_hour': int(off_peak_hour),
            'off_peak_load_kw': float(off_peak_load),
            'intra_daily_variation': float((peak_load - off_peak_load) / off_peak_load * 100)
        }
        
        # Weekday vs. weekend
        daily_data = self.train_df.copy()
        daily_data['is_weekend'] = daily_data[self.timestamp_col].dt.weekday >= 5
        
        weekday_mean = daily_data[~daily_data['is_weekend']][self.target_col].mean()
        weekend_mean = daily_data[daily_data['is_weekend']][self.target_col].mean()
        
        diag['weekday_vs_weekend'] = {
            'weekday_mean_load_kw': float(weekday_mean),
            'weekend_mean_load_kw': float(weekend_mean),
            'difference_kw': float(weekday_mean - weekend_mean),
            'percent_difference': float((weekday_mean - weekend_mean) / weekend_mean * 100)
        }
        
        # Overall statistics
        diag['overall_statistics'] = {
            'mean_load_kw': float(load_series.mean()),
            'std_load_kw': float(load_series.std()),
            'min_load_kw': float(load_series.min()),
            'max_load_kw': float(load_series.max()),
            'coefficient_of_variation': float(load_series.std() / load_series.mean())
        }
        
        # Data coverage
        diag['data_coverage'] = {
            'total_hours': int(len(self.train_df)),
            'date_range_start': str(self.train_df[self.timestamp_col].min()),
            'date_range_end': str(self.train_df[self.timestamp_col].max()),
            'missing_values': int(self.train_df[self.target_col].isna().sum())
        }
        
        self._diagnostics = diag
        return diag
    
    def save_diagnostics(self, filepath: str = None) -> str:
        """
        Save diagnostics to JSON file.
        
        Args:
            filepath: Path to save JSON; if None, uses default location
            
        Returns:
            Path to saved file
        """
        if not self._diagnostics:
            self.compute_diagnostics()
        
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'diagnostics.json')
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self._diagnostics, f, indent=2)
        
        return filepath
    
    def print_diagnostics(self) -> None:
        """Print human-readable diagnostics to console."""
        if not self._diagnostics:
            self.compute_diagnostics()
        
        diag = self._diagnostics
        
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS - TRAINING DATA DIAGNOSTICS")
        print("="*80)
        
        print("\nDATA COVERAGE:")
        print(f"  Total hours: {diag['data_coverage']['total_hours']:,}")
        print(f"  Date range: {diag['data_coverage']['date_range_start']} to "
              f"{diag['data_coverage']['date_range_end']}")
        print(f"  Missing values: {diag['data_coverage']['missing_values']}")
        
        print("\nOVERALL LOAD STATISTICS:")
        stats = diag['overall_statistics']
        print(f"  Mean: {stats['mean_load_kw']:,.0f} kW")
        print(f"  Std Dev: {stats['std_load_kw']:,.0f} kW")
        print(f"  Min: {stats['min_load_kw']:,.0f} kW")
        print(f"  Max: {stats['max_load_kw']:,.0f} kW")
        print(f"  Coefficient of Variation: {stats['coefficient_of_variation']:.3f}")
        
        print("\nSEASONAL PATTERNS (DAILY CYCLE):")
        seasonal = diag['seasonal_profiles']
        print(f"  Peak hour: {seasonal['peak_hour']:02d}:00 ({seasonal['peak_load_kw']:,.0f} kW)")
        print(f"  Off-peak hour: {seasonal['off_peak_hour']:02d}:00 ({seasonal['off_peak_load_kw']:,.0f} kW)")
        print(f"  Intra-daily variation: {seasonal['intra_daily_variation']:.1f}%")
        
        print("\nWEEKDAY VS. WEEKEND:")
        weekday = diag['weekday_vs_weekend']
        print(f"  Weekday mean: {weekday['weekday_mean_load_kw']:,.0f} kW")
        print(f"  Weekend mean: {weekday['weekend_mean_load_kw']:,.0f} kW")
        print(f"  Difference: {weekday['difference_kw']:,.0f} kW ({weekday['percent_difference']:+.1f}%)")
        
        print("\nAUTOCORRELATION (SEASONALITY):")
        acf = diag['autocorrelation']
        print(f"  24-hour lag (daily): {acf['24_hour_lag']:.4f}")
        print(f"  168-hour lag (weekly): {acf['168_hour_lag']:.4f}")
        
        print("\nROLLING STATISTICS:")
        print(f"  7-day rolling mean: {diag['rolling_mean_7day']['mean']:,.0f} ± "
              f"{diag['rolling_mean_7day']['std']:,.0f} kW")
        print(f"  30-day rolling mean: {diag['rolling_mean_30day']['mean']:,.0f} ± "
              f"{diag['rolling_mean_30day']['std']:,.0f} kW")
        
        print("\n" + "="*80)
    
    def _export_plot_data(self, plot_name: str, data: pd.DataFrame) -> str:
        """
        Export plot data as CSV for reproducibility.
        
        Args:
            plot_name: Name of plot (e.g., 'plot_a')
            data: DataFrame with plot data
            
        Returns:
            Path to saved CSV file
        """
        csv_path = os.path.join(self.output_dir, f'{plot_name}_data.csv')
        data.to_csv(csv_path, index=False)
        return csv_path


def run_eda_pipeline(
    train_df: pd.DataFrame,
    output_dir: str = 'outputs/plots/eda_plots',
    timestamp_col: str = 'timestamp',
    target_col: str = 'load_kw'
) -> Dict[str, Any]:
    """
    Run complete EDA pipeline.
    
    Args:
        train_df: Training data DataFrame
        output_dir: Output directory for plots and diagnostics
        timestamp_col: Name of timestamp column
        target_col: Name of target column
        
    Returns:
        Dictionary with plot paths and diagnostics
    """
    plotter = EDAPlotter(
        train_df=train_df,
        timestamp_col=timestamp_col,
        target_col=target_col,
        output_dir=output_dir
    )
    
    # Generate plots
    plot_paths = plotter.generate_all_plots()
    
    # Compute and save diagnostics
    diagnostics = plotter.compute_diagnostics()
    diag_path = plotter.save_diagnostics()
    
    # Print diagnostics
    plotter.print_diagnostics()
    
    return {
        'plot_paths': plot_paths,
        'diagnostics': diagnostics,
        'diagnostics_file': diag_path
    }
