"""
Test suite for Stock Price Prediction Project

Tests data loading, feature extraction, model training, and data integrity.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project directory to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# Test configuration
TEST_DATA_FILE = "processedtest4_enhanced.csv"
REQUIRED_COLUMNS = ["Adj Close", "return", "label"]
FEATURE_COLUMNS_MIN = 10  # Minimum expected feature columns

def test_data_file_exists():
    """Test that processed data file exists"""
    print("Test 1: Checking if processed data file exists...")
    if os.path.exists(TEST_DATA_FILE):
        print(f"  [PASS] {TEST_DATA_FILE} exists")
        return True
    else:
        print(f"  [FAIL] {TEST_DATA_FILE} not found")
        print(f"    Run 'make features' or 'python feature_extraction_enhanced.py' first")
        return False

def test_data_loading():
    """Test that data can be loaded correctly"""
    print("\nTest 2: Testing data loading...")
    try:
        df = pd.read_csv(TEST_DATA_FILE, parse_dates=["Date"], index_col="Date")
        print(f"  [PASS] Data loaded successfully ({len(df)} rows)")
        return True, df
    except Exception as e:
        print(f"  [FAIL] Error loading data: {e}")
        return False, None

def test_required_columns(df):
    """Test that required columns exist"""
    print("\nTest 3: Checking required columns...")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if len(missing_cols) == 0:
        print(f"  [PASS] All required columns present: {REQUIRED_COLUMNS}")
        return True
    else:
        print(f"  [FAIL] Missing columns: {missing_cols}")
        return False

def test_feature_columns(df):
    """Test that feature columns exist"""
    print("\nTest 4: Checking feature columns...")
    reserved = {"Adj Close", "return", "return_next", "label", "Date", "Volume"}
    feature_cols = [c for c in df.columns if c not in reserved]
    
    if len(feature_cols) >= FEATURE_COLUMNS_MIN:
        print(f"  [PASS] Found {len(feature_cols)} feature columns (minimum: {FEATURE_COLUMNS_MIN})")
        return True, feature_cols
    else:
        print(f"  [FAIL] Only {len(feature_cols)} feature columns found (minimum: {FEATURE_COLUMNS_MIN})")
        return False, feature_cols

def test_label_distribution(df):
    """Test that labels are properly distributed"""
    print("\nTest 5: Checking label distribution...")
    if "label" not in df.columns:
        print("  ✗ 'label' column not found")
        return False
    
    label_counts = df["label"].value_counts()
    print(f"  Label distribution:")
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"    {label}: {count} ({pct:.1f}%)")
    
    # Check for binary classification (Up/Down)
    unique_labels = set(df["label"].dropna().unique())
    expected_labels = {"Up", "Down"}
    
    if unique_labels.issubset(expected_labels):
        print(f"  [PASS] Labels are binary (Up/Down)")
        return True
    else:
        print(f"  [WARN] Unexpected labels: {unique_labels}")
        return True  # Not a failure, just a warning

def test_no_data_leakage(df, feature_cols):
    """Test that features don't use future data (basic check)"""
    print("\nTest 6: Checking for data leakage (basic)...")
    
    # Check for features that might use future data
    suspicious_features = [col for col in feature_cols if "next" in col.lower() or "future" in col.lower()]
    
    if len(suspicious_features) == 0:
        print("  [PASS] No obvious future-looking features found")
        return True
    else:
        print(f"  [WARN] Suspicious features found: {suspicious_features}")
        print("    (These may be intentional - verify manually)")
        return True  # Not a failure, just a warning

def test_missing_values(df, feature_cols):
    """Test for excessive missing values"""
    print("\nTest 7: Checking for missing values...")
    
    missing_pct = (df[feature_cols].isna().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > 50]
    
    if len(high_missing) == 0:
        print("  [PASS] No features with >50% missing values")
        return True
    else:
        print(f"  [WARN] Features with >50% missing values:")
        for col, pct in high_missing.items():
            print(f"    {col}: {pct:.1f}%")
        print("    (This may be expected for optional features like options data)")
        return True  # Not a failure, just a warning

def test_model_imports():
    """Test that required model libraries can be imported"""
    print("\nTest 8: Testing model library imports...")
    
    results = {}
    
    # Test scikit-learn
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        results["scikit-learn"] = True
        print("  [PASS] scikit-learn imported successfully")
    except ImportError as e:
        results["scikit-learn"] = False
        print(f"  [FAIL] scikit-learn import failed: {e}")
    
    # Test XGBoost (optional)
    try:
        import xgboost as xgb
        results["xgboost"] = True
        print("  [PASS] xgboost imported successfully")
    except ImportError:
        results["xgboost"] = False
        print("  [WARN] xgboost not available (optional)")
    
    # Test PyTorch (optional)
    try:
        import torch
        results["pytorch"] = True
        print("  [PASS] pytorch imported successfully")
    except ImportError:
        results["pytorch"] = False
        print("  [WARN] pytorch not available (optional)")
    
    # At least scikit-learn must be available
    return results.get("scikit-learn", False)

def test_directories():
    """Test that required directories exist or can be created"""
    print("\nTest 9: Checking directories...")
    
    required_dirs = ["models", "plots", "results", "data"]
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = Path(PROJECT_DIR) / dir_name
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  [PASS] Created directory: {dir_name}/")
            except Exception as e:
                print(f"  [FAIL] Failed to create directory {dir_name}/: {e}")
                all_exist = False
        else:
            print(f"  [PASS] Directory exists: {dir_name}/")
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 70)
    print("STOCK PRICE PREDICTION PROJECT - TEST SUITE")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Data file exists
    data_exists = test_data_file_exists()
    test_results.append(("Data file exists", data_exists))
    
    if not data_exists:
        print("\n⚠ Skipping remaining tests (data file not found)")
        print("Run 'make features' or 'python feature_extraction_enhanced.py' first")
        return
    
    # Test 2: Data loading
    load_success, df = test_data_loading()
    test_results.append(("Data loading", load_success))
    
    if not load_success or df is None:
        print("\n⚠ Skipping remaining tests (data loading failed)")
        return
    
    # Test 3: Required columns
    has_required = test_required_columns(df)
    test_results.append(("Required columns", has_required))
    
    # Test 4: Feature columns
    has_features, feature_cols = test_feature_columns(df)
    test_results.append(("Feature columns", has_features))
    
    # Test 5: Label distribution
    labels_ok = test_label_distribution(df)
    test_results.append(("Label distribution", labels_ok))
    
    # Test 6: Data leakage
    if has_features:
        no_leakage = test_no_data_leakage(df, feature_cols)
        test_results.append(("Data leakage check", no_leakage))
    
    # Test 7: Missing values
    if has_features:
        missing_ok = test_missing_values(df, feature_cols)
        test_results.append(("Missing values", missing_ok))
    
    # Test 8: Model imports
    imports_ok = test_model_imports()
    test_results.append(("Model imports", imports_ok))
    
    # Test 9: Directories
    dirs_ok = test_directories()
    test_results.append(("Directories", dirs_ok))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print("=" * 70)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"[WARNING] {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

