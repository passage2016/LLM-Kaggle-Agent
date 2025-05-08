import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
sample_sub = pd.read_csv('./data/sample_submission.csv')

# Preprocessing function
def preprocess_data(df, is_train=True):
    # Make a copy of the dataframe
    df = df.copy()
    
    # Feature engineering
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['dewpoint_diff'] = df['temparature'] - df['dewpoint']
    
    # Select features
    features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed',
                'temp_range', 'dewpoint_diff']
    
    if is_train:
        X = df[features]
        y = df['rainfall']
        return X, y
    else:
        return df[features]

# Preprocess train and test data
X_train, y_train = preprocess_data(train_df, is_train=True)
X_test = preprocess_data(test_df, is_train=False)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define neural network model
class RainfallPredictor(nn.Module):
    def __init__(self, input_size):
        super(RainfallPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize model
input_size = X_train.shape[1]
model = RainfallPredictor(input_size).to(device)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Make predictions with neural network
model.eval()
with torch.no_grad():
    nn_preds = model(X_test_tensor).cpu().numpy().flatten()

# Train XGBoost model as well
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Train LightGBM model
lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
lgbm_model.fit(X_train, y_train)
lgbm_preds = lgbm_model.predict(X_test)

# Ensemble predictions (simple average)
final_preds = (nn_preds + xgb_preds + lgbm_preds) / 3

# Create submission file
submission = sample_sub.copy()
submission['rainfall'] = final_preds

# Clip negative values to 0 (since rainfall can't be negative)
submission['rainfall'] = submission['rainfall'].clip(lower=0)

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")