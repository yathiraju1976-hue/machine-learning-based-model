# ADVANCED MACHINE LEARNING MODEL FOR DYE REMOVAL

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# -----------------------------
# STEP 1: LOAD DATASET
# -----------------------------
try:
    df = pd.read_excel("input.xlsx")
    print("Dataset loaded successfully!\n")
except:
    print("Error: input.xlsx file not found.")
    exit()

# -----------------------------
# STEP 2: CLEAN DATA
# -----------------------------
df.columns = df.columns.str.strip()

if "% Dye Removal Efficiency" in df.columns:
    df.rename(columns={"% Dye Removal Efficiency": "Dye Removal Efficiency"}, inplace=True)

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# -----------------------------
# STEP 3: FEATURES & TARGET
# -----------------------------
X = df[[
    "pH",
    "Dye Concentration (mg/L)",
    "Extract/Nanoparticle Dose (mg/100 mL)",
    "Contact Time (min)"
]]

y = df["Dye Removal Efficiency"]

# -----------------------------
# STEP 4: TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 5: RANDOM FOREST MODEL + TUNING
# -----------------------------
rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)

# -----------------------------
# STEP 6: EVALUATE MODEL
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance:")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# -----------------------------
# STEP 7: FEATURE IMPORTANCE
# -----------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance)

# -----------------------------
# STEP 8: SAVE MODEL
# -----------------------------
joblib.dump(model, "rf_dye_removal_model.pkl")
print("\nModel saved successfully!")

# -----------------------------
# STEP 9: PREDICTION SYSTEM
# -----------------------------
print("\n----- Predict Dye Removal Efficiency -----")

while True:
    try:
        ph = float(input("Enter pH: "))
        dye_conc = float(input("Enter Dye Concentration (mg/L): "))
        extract_conc = float(input("Enter Extract/Nanoparticle Dose (mg/100 mL): "))
        contact_time = float(input("Enter Contact Time (min): "))

        new_data = pd.DataFrame([[
            ph, dye_conc, extract_conc, contact_time
        ]], columns=X.columns)

        prediction = model.predict(new_data)
        print(f"\nPredicted Efficiency: {prediction[0]:.2f}%\n")

    except ValueError:
        print("Invalid input! Enter numeric values.\n")

    again = input("Predict again? (yes/no): ").lower()
    if again != "yes":
        break