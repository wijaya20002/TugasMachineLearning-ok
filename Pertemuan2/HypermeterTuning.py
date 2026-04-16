from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# ================= DATA =================
X = np.array([50, 70, 90, 110, 130]).reshape(-1, 1)
y = np.array([300, 420, 500, 600, 750])

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# ================= GRID SEARCH =================
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=2)

grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)