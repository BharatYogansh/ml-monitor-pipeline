"""
Trains the baseline housing-price model and writes two artifacts that the
rest of the pipeline depends on:
  - model.pkl            the trained sklearn model
  - reference_data.csv   the training distribution, used later as the
                          "ground truth" the drift detector compares
                          live traffic against
"""
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data_gen import FEATURE_COLUMNS, generate_housing_data


def main():
    df = generate_housing_data(n=3000, drift=False, seed=42)
    X, y = df[FEATURE_COLUMNS], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: ${mae:,.0f}")
    print(f"Test R2:  {r2:.3f}")

    joblib.dump(model, "model.pkl")
    df.to_csv("reference_data.csv", index=False)
    print("Saved model.pkl and reference_data.csv")


if __name__ == "__main__":
    main()
