# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

# Print all columns for debugging
print("All columns in train data:", train_data.columns.tolist())
print("All columns in test data:", test_data.columns.tolist())

# Convert all column names to lowercase for consistency
train_data.columns = train_data.columns.str.lower()
test_data.columns = test_data.columns.str.lower()

# Print column names to verify after lowercase conversion
print("\nColumns after lowercase conversion:")
print("Train columns:", train_data.columns.tolist())
print("Test columns:", test_data.columns.tolist())

# Identify the correct target and ID column names by checking the actual columns
# Common variations for target column in housing datasets
possible_targets = ['saleprice', 'price', 'sale_price', 'saleamount', 'target', 'value']
possible_ids = ['id', 'pid', 'parcelid', 'identifier']

# Find which columns actually exist in our data
target_column = next((col for col in possible_targets if col in train_data.columns), None)
id_column = next((col for col in possible_ids if col in train_data.columns), None)

# If target column not found, let's look for the last numeric column (common pattern)
if not target_column:
    print("\nWarning: Could not identify target column automatically")
    numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print("Available numeric columns:", numeric_cols)
    
    # Exclude possible ID columns
    candidate_cols = [col for col in numeric_cols if col not in possible_ids]
    
    if len(candidate_cols) == 1:
        target_column = candidate_cols[0]
        print(f"\nAutomatically selected '{target_column}' as target column")
    else:
        # If multiple candidates, pick the one that looks most like a target
        for col in candidate_cols:
            if 'price' in col or 'value' in col or 'amount' in col:
                target_column = col
                print(f"\nSelected '{target_column}' as target column based on name")
                break
        
        if not target_column:
            # As last resort, pick the last numeric column that's not ID
            target_column = candidate_cols[-1] if candidate_cols else None
            print(f"\nSelected '{target_column}' as target column (last numeric column)")
            
if not target_column:
    raise ValueError("\nError: Could not identify target column in the dataset. "
                    "Please specify manually by uncommenting and setting these variables:\n"
                    "# target_column = 'your_target_column_name'\n"
                    "# id_column = 'your_id_column_name'")

if not id_column:
    print("\nWarning: Could not identify ID column automatically")
    # Check for columns with unique values
    for col in train_data.columns:
        if train_data[col].nunique() == len(train_data):
            id_column = col
            print(f"\nSelected '{col}' as ID column (contains unique values)")
            break
    
    if not id_column and 'id' in train_data.columns:
        id_column = 'id'
        print(f"\nUsing 'id' as ID column")
    elif not id_column:
        id_column = train_data.columns[0]
        print(f"\nUsing first column '{id_column}' as ID column")

print(f"\nUsing target column: {target_column}")
print(f"Using ID column: {id_column}")

# Separate features and target
X = train_data.drop([id_column, target_column], axis=1)
y = train_data[target_column]
test_ids = test_data[id_column]
test_data = test_data.drop(id_column, axis=1)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data first
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_valid_preprocessed = preprocessor.transform(X_valid)

# Define XGBoost model with GPU support
xgb_model = xgb.XGBRegressor(
    tree_method='gpu_hist',
    gpu_id=0,
    random_state=42,
    n_estimators=1000,
    learning_rate=0.01,
    early_stopping_rounds=50
)

# Fit the model with preprocessed data
xgb_model.fit(X_train_preprocessed, y_train,
              eval_set=[(X_valid_preprocessed, y_valid)],
              verbose=False)

# Make predictions and calculate RMSE
xgb_preds = xgb_model.predict(X_valid_preprocessed)
xgb_rmse = np.sqrt(mean_squared_error(y_valid, xgb_preds))
print(f'XGBoost RMSE: {xgb_rmse:.4f}')

# Preprocess all training data for final model
X_preprocessed = preprocessor.fit_transform(X)
test_data_preprocessed = preprocessor.transform(test_data)

# Train final model on all training data
final_model = xgb.XGBRegressor(
    tree_method='gpu_hist',
    gpu_id=0,
    random_state=42,
    n_estimators=xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 1000,
    learning_rate=0.01
)

final_model.fit(X_preprocessed, y)

# Make predictions on test data
test_preds = final_model.predict(test_data_preprocessed)

# Create submission file (using original case for submission requirements)
output = pd.DataFrame({id_column.capitalize(): test_ids, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
print("Submission file created!")