# 1.Load training and testing data
# 2.scale the training data
# 3. save scaled data into processed folder

from sklearn.preprocessing import StandardScaler
import pandas as pd
from data_preprocessing import load_and_split_data
import pickle

# load data
X_train, X_test, y_train, y_test = load_and_split_data()

scaler = StandardScaler()

# scale data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# save processed data
pd.DataFrame(X_train_scaled).to_csv("../data/processed/X_train.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("../data/processed/X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv", index=False)

# save scaler
with open("../artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Feature engineering completed")
