import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np

def model_training(data: pd.DataFrame,
                   save_path: str):
    """
    Train an XGBoost model via GridSearch, then dump the best estimator
    straight to `save_path` (no renaming needed).
    """
    # shuffle
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # split inputs & label
    X_train = data.drop(columns=['label'])
    y_train = data['label']

    # build the XGBoost + grid
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    param_grid = {
        'learning_rate':     [0.05, 0.1],         # conservative, smooth training
        'max_depth':         [4, 5, 6],           # sufficient for feature interactions
        'n_estimators':      [200, 300],          # allows convergence
        'subsample':         [0.8, 1.0],          # avoid overfitting noise
        'colsample_bytree':  [0.8, 1.0],          # regularize feature usage
        'gamma':             [0, 1]               # discourage unhelpful splits
    }

    gs = GridSearchCV(
        model,
        param_grid,
        scoring='roc_auc',
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_

    # make sure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(best_model, save_path)

    print(f"Saved best model to {save_path}")
    return best_model