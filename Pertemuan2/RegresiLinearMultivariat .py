import pandas as pd
from sklearn.linear_model import LinearRegression

# Data contoh multivariat
data = {
    'Luas (m²)': [50, 70, 90, 110, 130],
    'Kamar': [2, 3, 3, 4, 4],
    'Jarak (km)': [10, 5, 3, 2, 1],
    'Harga (juta Rp)': [300, 420, 500, 600, 750]
}

df = pd.DataFrame(data)

# Pisahkan fitur (X) dan target (Y)
X = df[['Luas (m²)', 'Kamar', 'Jarak (km)']]
Y = df['Harga (juta Rp)']

# Latih model
model = LinearRegression()
model.fit(X, Y)

# Koefisien regresi
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Koefisien (β₁, β₂, β₃): {model.coef_}")

# Prediksi untuk rumah: 100m², 3 kamar, 4 km dari pusat kota
prediksi = model.predict([[100, 3, 4]])
print(f"Prediksi harga: Rp {prediksi[0]:.0f} juta")