import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================
# Generate Data
# =====================
np.random.seed(42)

data = {
    'luas_tanah': np.random.randint(50, 200, 100),
    'jumlah_kamar': np.random.randint(1, 5, 100),
    'jarak_ke_pusat_kota': np.random.uniform(1, 20, 100),
    'tahun_dibangun': np.random.randint(1990, 2023, 100),
}

df = pd.DataFrame(data)

df['harga'] = (
    5 * df['luas_tanah'] +
    50 * df['jumlah_kamar'] -
    10 * df['jarak_ke_pusat_kota'] -
    0.5 * (2023 - df['tahun_dibangun']) +
    np.random.normal(0, 50, 100)
)

# =====================
# Split Data
# =====================
X = df.drop('harga', axis=1)
y = df['harga']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# Train Model
# =====================
model = RandomForestRegressor()
model.fit(X_train, y_train)

# =====================
# Visualisasi Feature Importance
# =====================
importance = model.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.xlabel("Importance Score")
plt.title("Kontribusi Fitur terhadap Prediksi Harga")
plt.show()