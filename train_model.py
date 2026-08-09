"""
Trains the baseline model on the real Windsor Housing dataset and writes:
  - model.pkl            the trained sklearn model
  - reference_data.csv   the full real dataset, used later as the
                          "ground truth" the drift detector compares
                          live traffic against
"""
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from data_gen import FEATURE_COLUMNS, TARGET_COLUMN, load_real_data


def main():
    df = load_real_data()
    X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MAE: ${mae:,.0f}")
    print(f"Test R2:  {r2:.3f}")

    joblib.dump(model, "model.pkl")
    df.to_csv("reference_data.csv", index=False)
    print(f"Saved model.pkl and reference_data.csv ({len(df)} real rows, Windsor Housing dataset)")


if __name__ == "__main__":
    main()
