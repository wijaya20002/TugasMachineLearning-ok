from sklearn.metrics import r2_score
from EvaluasiModel import Y_test, Y_pred

r2 = r2_score(Y_test, Y_pred)
print(f"R² Score: {r2:.2f}")