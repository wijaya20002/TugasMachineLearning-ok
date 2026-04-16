from sklearn.metrics import mean_squared_error

# Data aktual (Y) dan prediksi (Y_pred)
Y = np.array([300, 420, 500, 600, 750])
Y_pred = model.predict(X)  # Misal: [320, 410, 490, 610, 740]

# Hitung MSE
mse = mean_squared_error(Y, Y_pred)
print(f"MSE: {mse:.2f}")