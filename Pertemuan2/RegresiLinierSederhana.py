import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data contoh: luas tanah (X) vs harga rumah (Y)
X = np.array([50, 70, 90, 110, 130]).reshape(-1, 1)  # Fitur (dalam m²)
Y = np.array([300, 420, 500, 600, 750])             # Target (dalam juta Rp)

# Latih model regresi linear
model = LinearRegression()
model.fit(X, Y)

# Prediksi harga untuk luas 100 m²
prediksi = model.predict(np.array([[100]]))
print(f"Prediksi harga untuk luas 100 m²: Rp {prediksi[0]:.0f} juta")

# Visualisasi
plt.scatter(X, Y, color='blue', label='Data Asli')
plt.plot(X, model.predict(X), color='red', label='Garis Regresi')
plt.xlabel('Luas Tanah (m²)')
plt.ylabel('Harga Rumah (juta Rp)')
plt.legend()
plt.show()