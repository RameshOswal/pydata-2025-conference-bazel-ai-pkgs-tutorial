"""
Simple test for data processing functionality
"""

import os
import sys
import tempfile
from data_processor import load_pumpkin_data, clean_and_filter_data, prepare_features


def test_data_processing():
    """Test the basic data processing pipeline."""
    print("Running data processing tests...")
    
    # Find the data file
    data_path = "examples/01-basic-ml-pipeline/data/US-pumpkins.csv"
    if not os.path.exists(data_path):
        # Try alternative path when run from Bazel
        data_path = "data/US-pumpkins.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Could not find data file at {data_path}")
        return False
    
    try:
        # Test data loading
        raw_data = load_pumpkin_data(data_path)
        assert len(raw_data) > 0, "No data loaded"
        print(f"✓ Data loading: {len(raw_data)} rows loaded")
        
        # Test data filtering
        filtered_data = clean_and_filter_data(raw_data)
        assert len(filtered_data) <= len(raw_data), "Filtering should not increase data size"
        print(f"✓ Data filtering: {len(filtered_data)} rows after filtering")
        
        # Test feature preparation
        processed_data = prepare_features(filtered_data)
        expected_columns = ['Month', 'Package', 'Low Price', 'High Price', 'Price']
        assert all(col in processed_data.columns for col in expected_columns), "Missing expected columns"
        print(f"✓ Feature preparation: {len(processed_data.columns)} columns prepared")
        
        print("All tests passed! ✓")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = test_data_processing()
    sys.exit(0 if success else 1)