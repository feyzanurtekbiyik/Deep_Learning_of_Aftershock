
import os

import pandas as pd

import numpy as np

from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization



# === 1. LOAD DATA ===

train_dir = "C:/Users/PC/OneDrive/Masaüstü/traindata"

test_dir = "C:/Users/PC/OneDrive/Masaüstü/testdata"



train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.csv')]

test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.csv')]



train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

test_df = pd.concat([pd.read_csv(f) for f in test_files], ignore_index=True)



# === 2. ADD ABSOLUTE VALUES OF COORDINATES ===

for df in [train_df, test_df]:

    df["abs_x"] = df["x"].abs()

    df["abs_y"] = df["y"].abs()

    df["abs_z"] = df["z"].abs()



# === 3. FEATURES ===

selected_columns = [

    "stresses_full_xx", "stresses_full_yy", "stresses_full_zz",

    "stresses_full_xy", "stresses_full_yz", "stresses_full_xz",

    "euclidean_distance", "abs_x", "abs_y", "abs_z"

]

target_column = "grid_aftershock_count"



# === 4. X and Y ===

X_train = train_df[selected_columns]

y_train = train_df[target_column]

X_test = test_df[selected_columns]

y_test = test_df[target_column]



# === 5. STANDARDIZATION ===

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



# === 6. Bayes Optimization ===

def xgb_evaluate(n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_alpha, reg_lambda):

    model = XGBRegressor(

        n_estimators=int(n_estimators),

        learning_rate=learning_rate,

        max_depth=int(max_depth),

        subsample=subsample,

        colsample_bytree=colsample_bytree,

        reg_alpha=reg_alpha,

        reg_lambda=reg_lambda,

        eval_metric="rmse",

        objective="reg:squarederror",

        random_state=42,

        verbosity=0

    )

    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)

    return -mean_squared_error(y_test, preds)




param_bounds = {

    "n_estimators": (100, 800),

    "learning_rate": (0.0005, 0.2),

    "max_depth": (3, 12),

    "subsample": (0.4, 1.0),

    "colsample_bytree": (0.4, 1.0),

    "reg_alpha": (0, 5),

    "reg_lambda": (0, 5)

}



optimizer = BayesianOptimization(

    f=xgb_evaluate,

    pbounds=param_bounds,

    random_state=42,

    verbose=2

)

optimizer.maximize(init_points=5, n_iter=20)



# === 7. TRAIN WITH BEST MODEL ===

best_params = optimizer.max["params"]

best_params["n_estimators"] = int(best_params["n_estimators"])

best_params["max_depth"] = int(best_params["max_depth"])



print("\n Best Parameters:", best_params)



best_model = XGBRegressor(

    n_estimators=best_params["n_estimators"],

    learning_rate=best_params["learning_rate"],

    max_depth=best_params["max_depth"],

    subsample=best_params["subsample"],

    colsample_bytree=best_params["colsample_bytree"],

    reg_alpha=best_params["reg_alpha"],

    reg_lambda=best_params["reg_lambda"],

    eval_metric="rmse",

    objective="reg:squarederror",

    tree_method='hist',            

    grow_policy='lossguide',       

    min_child_weight=5,           

    max_bin=256,                   

    random_state=42

)

best_model.fit(X_train_scaled, y_train)



# === 8.  Evaluation ===

y_pred = np.maximum(best_model.predict(X_test_scaled), 0)



mse = mean_squared_error(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)



print(f"\n📌 XGBoost Results:")

print(f"   MSE: {mse:.4f}")

print(f"   MAE: {mae:.4f}")

print(f"   R²: {r2:.4f}")



# === 9. Graphics ===

plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')

plt.xlabel('Actual')

plt.ylabel('Predicted')

plt.title('Actual vs Predicted Aftershock Count')

plt.grid(True)

plt.show()



plt.figure(figsize=(10, 6))

plt.hist(y_test - y_pred, bins=50, color='skyblue', edgecolor='black')

plt.xlabel("Error (Actual - Predictes)")

plt.title("Error Distribution")

plt.grid(True)

plt.show()

"""
   MSE: 0.0651

   MAE: 0.0548

   R²: 0.2374
"""
