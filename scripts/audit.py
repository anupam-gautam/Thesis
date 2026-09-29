"""Data audit CLI.

Usage:
    python scripts/audit.py <data_file_path>
    python scripts/audit.py DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader


def main():
    """Run data audit: load, validate, print summary."""
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/audit.py <data_file_path>")
        print("Example: python scripts/audit.py DataSet/CLEAN_smartgrid_5yr_hourly_natural.csv")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        loader = DataLoader(dataset_path)
        df = loader.load()
        loader.print_summary()
        
        # Get and display summary
        summary = loader.get_summary()
        print("\n✓ Data loading and schema validation complete")
        print(f"\nSummary Statistics:")
        print(f"  - File: {summary['file_path']}")
        print(f"  - Rows: {summary['row_count']}")
        print(f"  - Columns: {summary['column_count']}")
        print(f"  - Timestamp Column: {summary['timestamp_column']}")
        print(f"  - Target Column: {summary['target_column']}")
        if summary['timestamp_range']:
            print(f"  - Date Range: {summary['timestamp_range']['first']} to {summary['timestamp_range']['last']}")
        
    except Exception as e:
        print(f"✗ Error during data audit: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
