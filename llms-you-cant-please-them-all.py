# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import os

# Configure TensorFlow to use CPU if GPU setup fails
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info messages

# First check if TensorFlow is installed, if not install it
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping
    
    # Check GPU availability and configure properly
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# Display basic info about the data
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("\nTrain data columns:", train_df.columns.tolist())
print("\nTrain data head:")
print(train_df.head())

# Check the actual target column name - look for numeric columns that might represent listening time
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
target_col = [col for col in numeric_cols if 'listen' in col.lower() or 'time' in col.lower()]
if not target_col:
    # If no obvious target found, take the last numeric column as target
    target_col = numeric_cols[-1]
    print(f"\nWarning: Could not find obvious target column, using '{target_col}' as target variable")
else:
    target_col = target_col[0]
print(f"\nUsing '{target_col}' as target variable")

# Verify target column is numeric
if not np.issubdtype(train_df[target_col].dtype, np.number):
    raise ValueError(f"Target column '{target_col}' is not numeric. Please check your data.")

# Separate features and target
X = train_df.drop(['id', target_col], axis=1)
y = train_df[target_col]

# Get the feature columns from training data
features_cols = X.columns.tolist()

# Check if test data has 'id' column, if not add it (assuming it's the first column)
if 'id' not in test_df.columns:
    test_df.reset_index(inplace=True)
    test_df.rename(columns={'index': 'id'}, inplace=True)

# Only keep columns that exist in both training and test data
common_cols = [col for col in features_cols if col in test_df.columns]
X = X[common_cols]  # Update training features to only include common columns
X_test = test_df[common_cols]

# Identify categorical and numerical columns more carefully
numerical_cols = []
categorical_cols = []

for col in X.columns:
    # Try to convert to numeric - if successful, it's numerical
    try:
        pd.to_numeric(X[col])
        numerical_cols.append(col)
    except ValueError:
        categorical_cols.append(col)

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

# Convert numerical columns to float32 to ensure compatibility
for col in numerical_cols:
    X.loc[:, col] = pd.to_numeric(X[col], errors='coerce').astype('float32')
    X_test.loc[:, col] = pd.to_numeric(X_test[col], errors='coerce').astype('float32')

# Convert categorical columns to string type to ensure proper encoding
for col in categorical_cols:
    X.loc[:, col] = X[col].astype(str)
    X_test.loc[:, col] = X_test[col].astype(str)

# Preprocessing pipeline
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)

# Convert to float32 for TensorFlow compatibility
X_train_preprocessed = X_train_preprocessed.astype('float32')
X_val_preprocessed = X_val_preprocessed.astype('float32')
X_test_preprocessed = X_test_preprocessed.astype('float32')

# Get the number of features after preprocessing
num_features = X_train_preprocessed.shape[1]

# Build a neural network model
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),  # Explicit input layer
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

# Create a device strategy to handle GPU/CPU allocation
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Number of devices: {strategy.num_replicas_in_sync}')
    with strategy.scope():
        model = build_model()
except:
    print("Could not create MirroredStrategy, falling back to default strategy")
    model = build_model()

# Early stopping callback
early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)

# Train the model with fallback to CPU if GPU fails
try:
    history = model.fit(
        X_train_preprocessed, y_train.values.astype('float32'),
        validation_data=(X_val_preprocessed, y_val.values.astype('float32')),
        batch_size=32,
        epochs=100,
        callbacks=[early_stopping],
        verbose=1
    )
except Exception as e:
    print(f"GPU training failed with error: {e}")
    print("Falling back to CPU training...")
    with tf.device('/CPU:0'):
        model = build_model()  # Rebuild model for CPU
        history = model.fit(
            X_train_preprocessed, y_train.values.astype('float32'),
            validation_data=(X_val_preprocessed, y_val.values.astype('float32')),
            batch_size=32,
            epochs=100,
            callbacks=[early_stopping],
            verbose=1
        )

# Evaluate the model
try:
    val_preds = model.predict(X_val_preprocessed).flatten()
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {val_rmse:.4f}")
except Exception as e:
    print(f"Error during prediction: {e}")
    print("Trying prediction on CPU...")
    with tf.device('/CPU:0'):
        val_preds = model.predict(X_val_preprocessed).flatten()
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"Validation RMSE: {val_rmse:.4f}")

# Make predictions on test set
try:
    test_preds = model.predict(X_test_preprocessed).flatten()
except Exception as e:
    print(f"Error during test prediction: {e}")
    print("Trying test prediction on CPU...")
    with tf.device('/CPU:0'):
        test_preds = model.predict(X_test_preprocessed).flatten()

# Create submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': test_preds
})

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created successfully!")