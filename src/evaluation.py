"""Model definitions, training, hyperparameter tuning, evaluation, and plots.

Mirrors the analysis notebook:
- Four classifiers wrapped in leakage-safe Pipelines
- GridSearchCV with 5-fold Stratified CV, optimizing F1 (imbalanced target)
- Evaluation with Accuracy / Precision / Recall / F1 on a held-out test set
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = "f1"

PARAM_GRIDS = {
    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    },
    "Decision Tree": {
        "model__max_depth": [3, 5, 7, 10, None],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__class_weight": [None, "balanced"],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [1, 5],
        "model__class_weight": [None, "balanced"],
    },
    "KNN": {
        "model__n_neighbors": [5, 11, 21, 31],
        "model__weights": ["uniform", "distance"],
    },
}


def build_models(preprocessor) -> dict:
    """Four classifiers, each wrapped in a Pipeline with the preprocessor."""
    return {
        "Logistic Regression": Pipeline(
            [("preprocess", preprocessor),
             ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]
        ),
        "Decision Tree": Pipeline(
            [("preprocess", preprocessor),
             ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]
        ),
        "Random Forest": Pipeline(
            [("preprocess", preprocessor),
             ("model", RandomForestClassifier(random_state=RANDOM_STATE))]
        ),
        "KNN": Pipeline(
            [("preprocess", preprocessor),
             ("model", KNeighborsClassifier())]
        ),
    }


def evaluate_models(models: dict, X_train, X_test, y_train, y_test,
                    fit: bool = True) -> pd.DataFrame:
    """Fit (optionally) and evaluate each model on the test set."""
    rows = []
    for name, model in models.items():
        if fit:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
        })
    return pd.DataFrame(rows)


def tune_models(models: dict, X_train, y_train) -> dict:
    """GridSearchCV for every model. Returns {name: fitted GridSearchCV}."""
    searches = {}
    for name, pipeline in models.items():
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRIDS[name],
            scoring=SCORING,
            cv=CV,
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)
        searches[name] = gs
    return searches


def build_comparison_table(baseline_df: pd.DataFrame,
                           tuned_df: pd.DataFrame) -> pd.DataFrame:
    """Merge baseline and tuned results into one side-by-side table."""
    base = baseline_df.rename(columns={
        "Accuracy": "Baseline Accuracy", "Precision": "Baseline Precision",
        "Recall": "Baseline Recall", "F1-Score": "Baseline F1",
    })
    tuned = tuned_df.rename(columns={
        "Accuracy": "Tuned Accuracy", "Precision": "Tuned Precision",
        "Recall": "Tuned Recall", "F1-Score": "Tuned F1",
    })
    return base.merge(tuned, on="Model").sort_values("Tuned F1", ascending=False)


# ---------------------------- plots ----------------------------

def plot_model_comparison(results_df: pd.DataFrame, title: str, save_path=None):
    """Grouped bar chart of all metrics for all models."""
    long = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(11, 6))
    sns.barplot(data=long, x="Metric", y="Score", hue="Model")
    plt.title(title, fontsize=15, weight="bold")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(model, X_test, y_test, model_name: str, save_path=None):
    """Confusion matrix with programmatic (never-stale) annotations."""
    cm = confusion_matrix(y_test, model.predict(X_test))
    labels = np.array([
        [f"True Negative\n{cm[0, 0]}", f"False Positive\n{cm[0, 1]}"],
        [f"False Negative\n{cm[1, 0]}", f"True Positive\n{cm[1, 1]}"],
    ])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {model_name}", weight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_roc_curves(models: dict, X_test, y_test, save_path=None):
    """ROC curves for any number of fitted models on one axis."""
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.title("ROC Curve Comparison", weight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_importance(fitted_pipeline, top_n: int = 10, save_path=None):
    """Top-N feature importances for a fitted tree-based pipeline."""
    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = fitted_pipeline.named_steps["model"].feature_importances_
    top = (pd.DataFrame({"Feature": names, "Importance": importances})
           .sort_values("Importance", ascending=False).head(top_n))
    top["Feature"] = (top["Feature"].str.replace("cat__", "", regex=False)
                                    .str.replace("num__", "", regex=False))
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="Importance", y="Feature",
                hue="Feature", palette="tab10", legend=False)
    plt.title(f"Top {top_n} Features Influencing Customer Churn", weight="bold")
    plt.ylabel("")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ---------------------------- EDA plots ----------------------------

def plot_churn_distribution(df, save_path=None):
    """Countplot of the target variable (churn vs no churn)."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churn")
    plt.title("Churn Distribution", weight="bold")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_tenure_histogram(df, save_path=None):
    """Histogram (with KDE) of customer tenure."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["tenure"], bins=30, kde=True)
    plt.title("Distribution of Customer Tenure", weight="bold")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
