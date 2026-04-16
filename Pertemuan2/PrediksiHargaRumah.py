import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# Generate data rumah acak
# ===============================
np.random.seed(42)

data = {
    'luas_tanah': np.random.randint(50, 200, 100),
    'jumlah_kamar': np.random.randint(1, 5, 100),
    'jarak_ke_pusat_kota': np.round(np.random.uniform(1, 20, 100), 1),
    'tahun_dibangun': np.random.randint(1990, 2023, 100)
}

df = pd.DataFrame(data)

# ===============================
# Simulasi harga rumah
# ===============================
df['harga'] = (
    5 * df['luas_tanah'] +
    50 * df['jumlah_kamar'] -
    10 * df['jarak_ke_pusat_kota'] -
    0.5 * (2023 - df['tahun_dibangun']) +
    np.random.normal(0, 50, 100)
)

# ===============================
# Pisahkan fitur dan target
# ===============================
X = df[['luas_tanah', 'jumlah_kamar', 'jarak_ke_pusat_kota', 'tahun_dibangun']]
y = df['harga']

# ===============================
# Split data training & testing
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Buat dan latih model
# ===============================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Prediksi data test
# ===============================
y_pred = model.predict(X_test)

# ===============================
# Evaluasi model
# ===============================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("=== Evaluasi Model ===")
print("MSE  :", round(mse, 2))
print("RMSE :", round(rmse, 2))
print("R2   :", round(r2, 2))

# ===============================
# Prediksi data baru
# ===============================
data_baru = pd.DataFrame({
    'luas_tanah': [120],
    'jumlah_kamar': [3],
    'jarak_ke_pusat_kota': [5],
    'tahun_dibangun': [2015]
})

prediksi = model.predict(data_baru)

print("\n=== Prediksi Rumah Baru ===")
print("Prediksi Harga Rumah:", round(prediksi[0], 2))