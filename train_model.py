import json, datetime
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CAT = ["Fuel_Type", "Seller_Type", "Transmission"]
NUM = ["Present_Price", "Kms_Driven", "Owner", "Car_Age"]
FEATURES = NUM + CAT


def load_data(path="data/car_data.csv"):
    df = pd.read_csv(path).drop_duplicates().dropna()
    df["Car_Age"] = datetime.datetime.now().year - df["Year"]
    df = df[(df["Kms_Driven"] < 500000) & (df["Car_Age"] >= 0)]
    return df


def build(model):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)], remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])


def train(path="data/car_data.csv", out_model="model/car_price_model.pkl", out_metrics="model/metrics.json"):
    df = load_data(path)
    X, y = df[FEATURES], df["Selling_Price"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    results, best_name, best_pipe, best_r2 = {}, None, None, -np.inf
    for name, m in candidates.items():
        pipe = build(m).fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        r2 = r2_score(y_te, p)
        results[name] = {"R2": round(r2, 4),
                         "MAE": round(mean_absolute_error(y_te, p), 4),
                         "RMSE": round(float(np.sqrt(mean_squared_error(y_te, p))), 4)}
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    joblib.dump(best_pipe, out_model)
    with open(out_metrics, "w") as f:
        json.dump({"best_model": best_name, "results": results, "rows": len(df)}, f, indent=2)
    return best_name, results


if __name__ == "__main__":
    name, res = train()
    print("Best:", name)
    print(json.dumps(res, indent=2))
