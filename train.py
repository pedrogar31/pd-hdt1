import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from utils import rmse, save_json


MODEL_REGISTRY = {
"LinearRegression": LinearRegression,
"RandomForestRegressor": RandomForestRegressor,
"GradientBoostingRegressor": GradientBoostingRegressor,
}


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_splits(p):
    #
    X_train = pd.read_parquet(p["output"]["X_train"]).to_numpy()
    y_train = pd.read_parquet(p["output"]["y_train"]).iloc[:, 0].to_numpy()
    return X_train, y_train

def cv_score(model, X, y, seed, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmse_scores, r2_scores, mae_scores = [], [], []
    for tr, va in kf.split(X):
        model.fit(X[tr], y[tr])
        preds = model.predict(X[va])
        rmse_scores.append(rmse(y[va], preds))
        r2_scores.append(r2_score(y[va], preds))
        mae_scores.append(mean_absolute_error(y[va], preds))
    
    return {
        "rmse": float(np.mean(rmse_scores)),
        "r2": float(np.mean(r2_scores)),
        "mae": float(np.mean(mae_scores))
    }




def main():
    p = load_params()
    X_train, y_train = load_splits(p)


    metrics_by_model = {}
    best = {"name": None, "type": None, "params": None, "rmse": float("inf")}
    
    
    for m in p["models"]:
        name, m_type, m_params = m["name"], m["type"], m["params"] or {}
        cls = MODEL_REGISTRY[m_type]
        model = cls(**m_params)
        scores = cv_score(model, X_train, y_train, seed=p["seed"], n_splits=5)
        metrics_by_model[name] = {"type": m_type, "params": m_params, **scores}
        if scores["rmse"] < best["rmse"]:
            best = {"name": name, "type": m_type, "params": m_params, "rmse": scores["rmse"]}
    
    
    # Reentrenar el mejor en todo el train
    best_model = MODEL_REGISTRY[best["type"]](**(best["params"] or {}))
    best_model.fit(X_train, y_train)
    
    
    Path("models").mkdir(exist_ok=True)
    dump(best_model, p["output"]["best_model"]) # guarda modelo
    
    
    Path("artifacts").mkdir(exist_ok=True)
    save_json(metrics_by_model, p["output"]["metrics_by_model"]) # metricas por modelo




if __name__ == "__main__":
    main()
