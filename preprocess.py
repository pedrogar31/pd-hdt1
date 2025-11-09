import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    p = params
    
    dataset_path = Path(p["input"]["dataset_path"])
    df = pd.read_csv(dataset_path)
    
    # Drop columnas
    drop_cols = p["columns"].get("drop", []) or []
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    
    target_col = p["columns"]["target"]
    assert target_col in df.columns, f"No se encuentra la columna objetivo: {target_col}"
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Detectar tipos
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    # Pipelines de preprocesamiento
    num_pipeline = [
        ("imputer", SimpleImputer(strategy=p["preprocess"]["impute_strategy_num"]))
    ]
    if p["preprocess"].get("scale_numeric", True):
        num_pipeline.append(("scaler", StandardScaler()))
    
    cat_pipeline = [
        ("imputer", SimpleImputer(strategy=p["preprocess"]["impute_strategy_cat"])) ,
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop=p["preprocess"]["one_hot_drop"],sparse_output=False))
    ]
    
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_pipeline), num_cols),
            ("cat", Pipeline(cat_pipeline), cat_cols)
        ], remainder="drop"
    )
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=p["preprocess"]["test_size"], random_state=p["seed"]
    )
    
    
    # Fit + transform
    full_pipeline = Pipeline([("pre", pre)])
    X_train_t = full_pipeline.fit_transform(X_train)
    X_test_t  = full_pipeline.transform(X_test)

    # Nombres de columnas resultantes
    feat_names = full_pipeline.named_steps["pre"].get_feature_names_out()

    # Asegurar carpetas
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/splits").mkdir(parents=True, exist_ok=True)

    # Guardados 
    pd.DataFrame(X_train_t, columns=feat_names, index=X_train.index).to_parquet(p["output"]["X_train"])
    pd.DataFrame(X_test_t,  columns=feat_names, index=X_test.index).to_parquet(p["output"]["X_test"])
    y_train.to_frame(name=target_col).to_parquet(p["output"]["y_train"])
    y_test.to_frame(name=target_col).to_parquet(p["output"]["y_test"])

    # Guardar dataset original procesado
    df.to_parquet(p["output"]["processed"])


if __name__ == "__main__":
    main()