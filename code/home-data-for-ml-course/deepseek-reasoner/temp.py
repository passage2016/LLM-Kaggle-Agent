import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Show current working directory first
print(f"Current working directory: {os.getcwd()}\n")

# Enhanced path detection with additional common paths and error handling
possible_paths = [
    # Kaggle default paths
    '/kaggle/input/house-prices-advanced-regression-techniques',
    '/kaggle/input/home-data-for-ml-course',
    '/kaggle/input/house-prices',
    '/kaggle/input',
    
    # Google Colab paths
    '/content',
    '/content/house-prices-advanced-regression-techniques',
    '/content/sample_data',
    
    # Common variations with potential typos
    '../input/house-prices-advanced-regression-techniques',
    '../input/home-data-for-ml-course',
    '../input/house-prices',
    '../input',
    'input',
    'data',
    '../data',
    
    # Local development paths
    os.path.join(os.getcwd(), 'input'),
    os.path.join(os.getcwd(), 'data'),
    os.path.join(os.getcwd(), 'dataset'),
    os.path.join(os.path.dirname(os.getcwd()), 'input'),
    
    # Competition specific variations
    '/kaggle/input/house-prices-advanced-regression-techniques-v2',
    '/kaggle/input/home-data-for-ml-course-v2',
    
    # Fallback paths
    '.',
    '..',
    '/kaggle/working',
    
    # Additional common paths
    '/kaggle/input/house-prices-data',
    '/kaggle/input/housing-data',
    '/kaggle/input/house-price-prediction',
    '/data',
    '/dataset'
]

# Expanded filename variations with case-insensitive matching
exact_matches = [
    ('train.csv', 'test.csv'),
    ('train_data.csv', 'test_data.csv'),
    ('housing_train.csv', 'housing_test.csv'),
    ('house_train.csv', 'house_test.csv'),
    ('train.csv.zip', 'test.csv.zip'),
    ('train.tsv', 'test.tsv'),
    ('train.parquet', 'test.parquet'),
    ('train_house.csv', 'test_house.csv'),
    ('houseprices_train.csv', 'houseprices_test.csv'),
    ('train_processed.csv', 'test_processed.csv')
]

found = False
checked_locations = []
train_file = None
test_file = None
data_dir = None

def validate_csv_pair(train_path, test_path):
    try:
        train_sample = pd.read_csv(train_path, nrows=5)
        train_sample.columns = train_sample.columns.str.strip().str.lower()
        if 'saleprice' not in train_sample.columns:
            checked_locations.append(f"Validation failed: 'saleprice' column not found in {train_path}")
            return False
        
        test_sample = pd.read_csv(test_path, nrows=5)
        test_sample.columns = test_sample.columns.str.strip().str.lower()
        if 'id' not in test_sample.columns:
            checked_locations.append(f"Validation failed: 'id' column not found in {test_path}")
            return False
            
        return True
    except Exception as e:
        checked_locations.append(f"Validation failed for {train_path} | {test_path}: {str(e)}")
        return False

# Enhanced case-insensitive path checking
def find_case_insensitive_path(directory, target_filename):
    target_lower = target_filename.lower()
    for filename in os.listdir(directory):
        if filename.lower() == target_lower:
            return os.path.join(directory, filename)
    return None

# First pass: Check exact path matches with case-insensitive search
for base_path in possible_paths:
    abs_path = os.path.abspath(base_path)
    if not os.path.exists(abs_path):
        checked_locations.append(f"{abs_path} (Path not found)")
        continue
    
    for train_name, test_name in exact_matches:
        train_path = find_case_insensitive_path(abs_path, train_name)
        test_path = find_case_insensitive_path(abs_path, test_name)
        
        if train_path and test_path:
            if validate_csv_pair(train_path, test_path):
                data_dir = abs_path
                train_file = os.path.basename(train_path)
                test_file = os.path.basename(test_path)
                found = True
                print(f"Found valid dataset in: {data_dir}")
                print(f"Train file: {train_file}")
                print(f"Test file: {test_file}")
                break
    if found:
        break

# Second pass: Deep search with relaxed criteria
if not found:
    print("\nPerforming deep search with relaxed criteria...")
    for base_path in possible_paths:
        abs_path = os.path.abspath(base_path)
        if not os.path.exists(abs_path):
            continue
            
        try:
            for root, dirs, files in os.walk(abs_path):
                csv_files = [f for f in files if f.lower().endswith(('.csv', '.tsv', '.parquet'))]
                train_candidates = [f for f in csv_files if any(kw in f.lower() for kw in ['train', 'training'])]
                test_candidates = [f for f in csv_files if any(kw in f.lower() for kw in ['test', 'testing', 'val', 'holdout'])]
                
                # Prioritize likely candidates
                train_candidates.sort(key=lambda x: (
                    -os.path.getsize(os.path.join(root, x)),
                    any(ex in x.lower() for ex in ['house', 'price', 'sale']),
                    x.lower()
                )
                
                test_candidates.sort(key=lambda x: (
                    -os.path.getsize(os.path.join(root, x)),
                    any(ex in x.lower() for ex in ['house', 'price']),
                    x.lower()
                )
                
                for t_train in train_candidates:
                    train_path = os.path.join(root, t_train)
                    for t_test in test_candidates:
                        test_path = os.path.join(root, t_test)
                        if validate_csv_pair(train_path, test_path):
                            data_dir = root
                            train_file = t_train
                            test_file = t_test
                            found = True
                            print(f"\nFound valid dataset in: {data_dir}")
                            print(f"Train file: {train_file}")
                            print(f"Test file: {test_file}")
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        except Exception as e:
            checked_locations.append(f"Error walking {abs_path}: {str(e)}")
            continue

# Final fallback: Check current directory directly
if not found:
    print("\nChecking current directory for files...")
    current_dir = os.getcwd()
    try:
        candidates = [f for f in os.listdir(current_dir) if f.lower().endswith(('.csv', '.tsv', '.parquet'))]
        train_candidates = [f for f in candidates if 'train' in f.lower()]
        test_candidates = [f for f in candidates if 'test' in f.lower()]
        
        for t_train in train_candidates:
            for t_test in test_candidates:
                train_path = os.path.join(current_dir, t_train)
                test_path = os.path.join(current_dir, t_test)
                if validate_csv_pair(train_path, test_path):
                    data_dir = current_dir
                    train_file = t_train
                    test_file = t_test
                    found = True
                    print(f"Found dataset in current directory: {current_dir}")
                    break
            if found:
                break

# Ultimate fallback: Manual input
if not found:
    print("\nERROR: Automatic detection failed. Last checked locations:")
    print('\n'.join(f"- {loc}" for loc in checked_locations[-20:]))  # Show last 20 errors
    
    print("\nCurrent directory structure:")
    os.system('ls -laR ./')
    
    # Try one last manual check with expanded patterns
    print("\nAttempting manual recovery with expanded patterns...")
    from pathlib import Path
    
    search_patterns = ['*train*', '*price*', '*house*', '*sale*']
    for pattern in search_patterns:
        for f in Path('.').rglob(pattern):
            if f.suffix.lower() in ['.csv', '.tsv', '.parquet']:
                try:
                    df = pd.read_csv(f, nrows=5)
                    df.columns = df.columns.str.strip().str.lower()
                    if 'saleprice' in df.columns:
                        print(f"\nPossible train file found: {f.resolve()}")
                        train_path = f.resolve()
                        # Look for corresponding test file
                        test_pattern = str(f.name).lower().replace('train', 'test')
                        test_candidates = list(f.parent.glob(test_pattern))
                        if not test_candidates:
                            test_pattern = str(f.name).lower().replace('training', 'testing')
                            test_candidates = list(f.parent.glob(test_pattern))
                        for t_test in test_candidates:
                            test_path = t_test.resolve()
                            if validate_csv_pair(train_path, test_path):
                                data_dir = str(f.parent)
                                train_file = f.name
                                test_file = t_test.name
                                found = True
                                print(f"Found matching test file: {test_path}")
                                break
                        if found:
                            break
                except Exception as e:
                    continue
        if found:
            break

    if not found:
        raise FileNotFoundError(
            "Failed to find dataset. Diagnostic information:\n"
            "1. Verify files exist in one of these paths:\n   " + 
            '\n   '.join(possible_paths) +
            "\n2. Check files have correct naming pattern\n"
            "3. Validate CSV files contain 'SalePrice' column in training data\n"
            "4. Files should be in CSV/TSV/Parquet format\n"
            "5. Check case sensitivity of filenames"
        )

# Load data with enhanced validation
try:
    print(f"\nLoading data from: {data_dir}")
    train = pd.read_csv(os.path.join(data_dir, train_file))
    test = pd.read_csv(os.path.join(data_dir, test_file))
    
    # Standardize column names
    train.columns = train.columns.str.strip().str.lower()
    test.columns = test.columns.str.strip().str.lower()
    
    print("\nData loaded successfully:")
    print(f"Training set shape: {train.shape}, Test set shape: {test.shape}")
    print("Sample train columns:", train.columns.tolist()[:10])
    print("Sample test columns:", test.columns.tolist()[:10])
    
    # Verify critical columns
    assert 'saleprice' in train.columns, "SalePrice column missing in training data"
    assert 'id' in test.columns, "ID column missing in test data"

except Exception as e:
    print(f"\nData validation failed: {str(e)}")
    print("First 5 rows of train:")
    print(train.head())
    print("\nFirst 5 rows of test:")
    print(test.head())
    raise

# Continue with your ML pipeline...
# [Add your data preprocessing and model training code here]