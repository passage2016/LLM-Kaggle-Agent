
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.datasets import fetch_california_housing

california = fetch_california_housing(as_frame=True)
data = california.frame
data.rename(columns={"MedHouseVal": "target"}, inplace=True)
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)
predictor = TabularPredictor(label="target").fit(train_data)
performance = predictor.evaluate(test_data)
print("Performance:", performance)
