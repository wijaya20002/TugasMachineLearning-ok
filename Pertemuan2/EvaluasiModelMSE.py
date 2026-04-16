import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Data
X = np.array([50, 70, 90, 110, 130]).reshape(-1, 1)
y = np.array([300, 420, 500, 600, 750])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"""
METRIK EVALUASI:
- MSE  : {mse:.2f}
- RMSE : {rmse:.2f}
- R2   : {r2:.2f}
- MAE  : {mae:.2f}
""")