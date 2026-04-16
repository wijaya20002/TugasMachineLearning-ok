from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Contoh data
data = {
    'luas': [50, 60, 70, 80, 90],
    'kamar': [2, 2, 3, 3, 4],
    'harga': [500, 600, 700, 800, 900]
}

df = pd.DataFrame(data)

# fitur dan target
X = df[['luas', 'kamar']]
y = df['harga']

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# preprocessing
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)

# model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# training
model.fit(X_train_processed, y_train)

print("Model berhasil dilatih")