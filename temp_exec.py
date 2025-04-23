# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Load the data
train_df = pd.read_csv('./data/dataset.csv')
test_df = pd.read_csv('./data/test.csv')
sample_submission = pd.read_csv('./data/sample_submission.csv')

# Basic data exploration (without visualization)
print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTraining data columns:\n", train_df.columns)
print("\nMissing values in training data:\n", train_df.isnull().sum())

# Preprocessing steps
# Separate features and target
X = train_df.drop(columns=['id', 'sale_price'])
y = train_df['sale_price']

# Identify categorical and numerical columns
categorical_cols = [cname for cname in X.columns if 
                    X[cname].dtype == 'object' and 
                    X[cname].nunique() < 50]
numerical_cols = [cname for cname in X.columns if 
                  X[cname].dtype in ['int64', 'float64'] and
                  cname not in categorical_cols]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=42)

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
mae = mean_absolute_error(y_valid, preds)
print(f"Validation MAE: {mae:,.0f}")

# Prepare test data (excluding id column)
test_X = test_df.drop(columns=['id'])

# Make predictions on test data
test_preds = my_pipeline.predict(test_X)

# Create prediction intervals (simple approach)
# Using standard deviation of training predictions as interval width
train_preds = my_pipeline.predict(X_train)
std_dev = np.std(train_preds)
lower_bound = test_preds - 1.96 * std_dev
upper_bound = test_preds + 1.96 * std_dev

# Ensure bounds are reasonable
lower_bound = np.maximum(lower_bound, 0)
upper_bound = np.minimum(upper_bound, y.max())

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'pi_lower': lower_bound,
    'pi_upper': upper_bound
})

# Save submission file
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"Submission file saved as {submission_file}")