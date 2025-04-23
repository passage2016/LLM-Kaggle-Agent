import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# Feature engineering function
def preprocess_data(df):
    # Extract features from Cabin
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    
    # Extract group and passenger number from PassengerId
    df[['Group', 'Passenger']] = df['PassengerId'].str.split('_', expand=True)
    df['Group'] = df['Group'].astype(int)
    df['Passenger'] = df['Passenger'].astype(int)
    
    # Calculate total spending
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    
    # Create family size feature from Name
    df['LastName'] = df['Name'].str.split().str[-1]
    df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count')
    
    # Drop columns that won't be used
    df = df.drop(['PassengerId', 'Cabin', 'Name', 'LastName'], axis=1)
    
    return df

# Preprocess data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Separate features and target
X = train_df.drop('Transported', axis=1)
y = train_df['Transported']
X_test = test_df.copy()

# Identify categorical and numerical columns
cat_cols = [cname for cname in X.columns if X[cname].dtype == 'object']
num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', LabelEncoder())
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Define model
model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=9)

# Create and evaluate pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Split data for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Make predictions
preds = pipeline.predict(X_valid)

# Evaluate model
print(f"Validation Accuracy: {accuracy_score(y_valid, preds):.4f}")

# Retrain on full training data
pipeline.fit(X, y)

# Make test predictions
test_preds = pipeline.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    'PassengerId': test_df.index,
    'Transported': test_preds
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created!")