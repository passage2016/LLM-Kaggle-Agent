import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# Basic data exploration
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTrain columns:", train_df.columns.tolist())
print("\nMissing values in train data:\n", train_df.isna().sum())
print("\nData types:\n", train_df.dtypes)

# Preprocessing pipeline
# Separate features and target
X = train_df.drop(['id', 'Listening_Time_minutes'], axis=1)  # Fixed column name
y = train_df['Listening_Time_minutes']  # Fixed column name

# Identify categorical and numerical features
categorical_features = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
numerical_features = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']

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
        ('num', numerical_transformer, numerical_features),  # Fixed typo in variable name
        ('cat', categorical_transformer, categorical_features)])

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

# Evaluate each model
results = {}
for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_valid)
    
    # Calculate metrics
    mse = mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)
    
    results[name] = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'R2': r2
    }
    
    print(f"\n{name} Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

# Hyperparameter tuning for the best model
best_model_name = max(results, key=lambda x: results[x]['R2'])
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

if best_model_name == 'RandomForest':
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    }
else:  # GradientBoosting
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth': [3, 5]
    }

# Create pipeline for grid search
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', models[best_model_name])])

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.2f}")

# Evaluate on validation set with best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_valid)

final_mse = mean_squared_error(y_valid, y_pred)
final_mae = mean_absolute_error(y_valid, y_pred)
final_r2 = r2_score(y_valid, y_pred)

print("\nFinal Model Performance:")
print(f"MSE: {final_mse:.2f}")
print(f"RMSE: {np.sqrt(final_mse):.2f}")
print(f"MAE: {final_mae:.2f}")
print(f"R2 Score: {final_r2:.2f}")

# Prepare test data for submission
test_ids = test_df['id']
X_test = test_df.drop(['id'], axis=1)

# Make predictions on test data
test_predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'Listening_Time_minutes': test_predictions
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("\nSubmission file created successfully!")