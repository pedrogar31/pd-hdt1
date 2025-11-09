import yaml
import pandas as pd
from joblib import load
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from utils import rmse, save_json




def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)




def main():
    p = load_params()
    
    
    X_test = pd.read_parquet(p["output"]["X_test"]).to_numpy()
    y_test = pd.read_parquet(p["output"]["y_test"]).iloc[:,0].to_numpy()
    
    
    model = load(p["output"]["best_model"])
    y_pred = model.predict(X_test)
    
    
    metrics = {
    "r2": r2_score(y_test, y_pred),
    "rmse": rmse(y_test, y_pred),
    "mae": mean_absolute_error(y_test, y_pred)
    }
    
    
    Path("artifacts").mkdir(exist_ok=True)
    save_json(metrics, p["output"]["metrics_global"]) # metricas generales
    
    
    # Guardar preds
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(p["output"]["preds"], index=False)




if __name__ == "__main__":
    main()