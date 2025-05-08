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

    # Split features and labels (convert to numpy)
    X_train = data.drop(columns=['label']).values
    y_train = data['label'].values

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



# in analysis_tools/BDT_tools/BDT_training.py

import matplotlib.pyplot as plt

def plot_feature_importances(
    model,
    feature_names: list,
    top_n: int = None,
    title: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Plot and (optionally) save a horizontal bar chart of feature importances.
    Shows the plot inline (for notebooks) and returns the Figure.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    if top_n is not None:
        indices = indices[:top_n]

    names  = [feature_names[i] for i in indices]
    values = importances[indices]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, max(4, len(names)*0.4)))

    # Plot horizontal bar chart
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()         # highest importance at top
    ax.set_xlabel("Feature importance")
    if title:
        ax.set_title(title)

    # Give extra room on the left for long labels
    fig.subplots_adjust(left=0.35, right=0.95, top=0.9, bottom=0.1)

    # Apply tight layout
    fig.tight_layout()

    # Save before closing
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')

    # Show inline in notebooks
    plt.show()

    return fig