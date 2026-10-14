"""Train and evaluate a profit model for the 50 Startups data set.

Outputs are written next to this file:
  startup_model.pkl.gz, evaluation_metrics.json, learning_curve.csv,
  learning_curve.png, actual_vs_predicted.png, feature_selection_results.csv,
  and feature_selection_metrics.png.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    cross_validate,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("state", OneHotEncoder(handle_unknown="ignore"), ["State"])],
        remainder="passthrough",
    )


def make_model(params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    preprocessor = make_preprocessor()
    regressor = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, **params
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def make_selection_model(params: dict[str, Any], k: int | str) -> Pipeline:
    """Build a leak-free pipeline: encode, select, then regress."""
    preprocessor = make_preprocessor()
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("selector", SelectKBest(score_func=f_regression, k=k)),
            (
                "regressor",
                RandomForestRegressor(
                    random_state=RANDOM_STATE, n_jobs=-1, **params
                ),
            ),
        ]
    )


def optimize(X: pd.DataFrame, y: pd.Series, trials: int) -> optuna.Study:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.7, 1.0]
            ),
        }
        scores = cross_val_score(
            make_model(params), X, y, scoring="neg_root_mean_squared_error", cv=cv
        )
        return float(-scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return study


def save_learning_curve(
    model: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, Any]:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    sizes, train_scores, valid_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_rmse, valid_rmse = -train_scores, -valid_scores
    frame = pd.DataFrame(
        {
            "training_examples": sizes,
            "train_rmse_mean": train_rmse.mean(axis=1),
            "train_rmse_std": train_rmse.std(axis=1),
            "validation_rmse_mean": valid_rmse.mean(axis=1),
            "validation_rmse_std": valid_rmse.std(axis=1),
        }
    )
    frame.to_csv(ROOT / "learning_curve.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, frame.train_rmse_mean, "o-", label="Training RMSE")
    plt.plot(sizes, frame.validation_rmse_mean, "o-", label="Validation RMSE")
    plt.fill_between(sizes, frame.train_rmse_mean-train_rmse.std(axis=1),
                     frame.train_rmse_mean+train_rmse.std(axis=1), alpha=.15)
    plt.fill_between(sizes, frame.validation_rmse_mean-valid_rmse.std(axis=1),
                     frame.validation_rmse_mean+valid_rmse.std(axis=1), alpha=.15)
    plt.xlabel("Training examples")
    plt.ylabel("RMSE")
    plt.title("Learning Curve")
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "learning_curve.png", dpi=150)
    plt.close()
    return frame.iloc[-1].to_dict()


def save_feature_selection_evaluation(
    params: dict[str, Any], X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Cross-validate feature counts and save their scores, names, and plot."""
    probe = make_selection_model(params, "all").fit(X, y)
    encoded_names = probe.named_steps["preprocessor"].get_feature_names_out()
    total_features = len(encoded_names)
    feature_counts = list(range(1, min(4, total_features) + 1))
    if total_features not in feature_counts:
        feature_counts.append(total_features)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for count in feature_counts:
        k = "all" if count == total_features else count
        model = make_selection_model(params, k)
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=-1,
        )

        # Fit once on all data solely to report the names selected at this k.
        model.fit(X, y)
        selected = encoded_names[model.named_steps["selector"].get_support()]
        selected = [name.split("__", 1)[-1] for name in selected]
        rows.append(
            {
                "Number of Features": count,
                "Selected Features": "; ".join(selected),
                "RMSE": float(-scores["test_rmse"].mean()),
                "R-squared": float(scores["test_r2"].mean()),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "feature_selection_results.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(results["Number of Features"], results["RMSE"], "o-", linewidth=2)
    axes[0].set_title("RMSE by Number of Features")
    axes[0].set_xlabel("Number of Features")
    axes[0].set_ylabel("RMSE")
    axes[1].plot(
        results["Number of Features"], results["R-squared"], "o-", linewidth=2
    )
    axes[1].set_title("R-squared by Number of Features")
    axes[1].set_xlabel("Number of Features")
    axes[1].set_ylabel("R-squared")
    for axis in axes:
        axis.set_xticks(feature_counts)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "feature_selection_metrics.png", dpi=150)
    plt.close(fig)
    return results


def seeded_mutual_information(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Deterministic score function for SelectKBest."""
    return mutual_info_regression(X, y, random_state=RANDOM_STATE)


def save_model_comparison(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate common regressors on one split and plot full-data predictions."""
    models = {
        "Linear Regression": [LinearRegression()],
        "Ridge": [Ridge(alpha=1.0)],
        "Lasso": [Lasso(alpha=0.1, max_iter=20000)],
        "Elastic Net": [ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000)],
        "SVR": [SVR(C=100.0, epsilon=0.1)],
        "Random Forest Regressor": [RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1, **params
        )],
        "Gradient Boosting Regressor": [GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )],
        "PCA + Linear Regression": [PCA(n_components=0.95), LinearRegression()],
        "Mutual Information + Linear Regression": [
            SelectKBest(score_func=seeded_mutual_information, k=3),
            LinearRegression(),
        ],
    }
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = [
            LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1, n_estimators=100)
        ]
    except ImportError:
        print("LightGBM is not installed; skipping it gracefully.")

    rows = []
    fitted_for_plot = {}
    for name, model_steps in models.items():
        steps = [("preprocessor", make_preprocessor()), ("scaler", StandardScaler())]
        if name == "PCA + Linear Regression":
            steps.extend([("pca", model_steps[0]), ("regressor", model_steps[1])])
        elif name == "Mutual Information + Linear Regression":
            steps.extend([("selector", model_steps[0]), ("regressor", model_steps[1])])
        else:
            steps.append(("regressor", model_steps[0]))
        pipeline = Pipeline(steps).fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        mse = float(mean_squared_error(y_test, prediction))

        encoded_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        clean_names = np.array([value.split("__", 1)[-1] for value in encoded_names])
        if "selector" in pipeline.named_steps:
            feature_description = "; ".join(
                clean_names[pipeline.named_steps["selector"].get_support()]
            )
        elif "pca" in pipeline.named_steps:
            components = pipeline.named_steps["pca"].n_components_
            feature_description = f"{components} PCA components (95% variance)"
        else:
            feature_description = "; ".join(clean_names)
        rows.append(
            {
                "Model": name,
                "Features used": feature_description,
                "MSE": mse,
                "RMSE": mse**0.5,
                "R-squared": float(r2_score(y_test, prediction)),
            }
        )
        fitted_for_plot[name] = clone(pipeline).fit(X, y)

    results = pd.DataFrame(rows).sort_values("RMSE", ignore_index=True)
    results.to_csv(ROOT / "model_comparison_metrics.csv", index=False)

    order = np.argsort(X["R&D Spend"].to_numpy())
    sorted_X = X.iloc[order]
    plt.figure(figsize=(13, 8))
    plt.scatter(X["R&D Spend"], y, color="black", s=38, label="Actual Data", zorder=10)
    for name, pipeline in fitted_for_plot.items():
        plt.plot(
            sorted_X["R&D Spend"],
            pipeline.predict(sorted_X),
            linewidth=1.6,
            alpha=0.85,
            label=name,
        )
    plt.title("Comparison of Different Regression Models")
    plt.xlabel("R&D Spend")
    plt.ylabel("Profit")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(ROOT / "model_comparison.png", dpi=150)
    plt.close()
    return results


def main(trials: int) -> None:
    data = pd.read_csv(ROOT / "50_Startups.csv")
    if data.isna().any().any():
        raise ValueError("Dataset contains missing values")
    X, y = data.drop(columns="Profit"), data["Profit"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    study = optimize(X_train, y_train, trials)
    model = make_model(study.best_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "test_r2": float(r2_score(y_test, predictions)),
        "optuna_best_cv_rmse": float(study.best_value),
        "optuna_trials": trials,
        "best_parameters": study.best_params,
    }
    metrics["learning_curve_full_data"] = save_learning_curve(model, X, y)
    selection_results = save_feature_selection_evaluation(study.best_params, X, y)
    best_rmse = selection_results.loc[selection_results["RMSE"].idxmin()]
    best_r2 = selection_results.loc[selection_results["R-squared"].idxmax()]
    metrics["feature_selection"] = {
        "lowest_rmse_feature_count": int(best_rmse["Number of Features"]),
        "lowest_rmse": float(best_rmse["RMSE"]),
        "highest_r2_feature_count": int(best_r2["Number of Features"]),
        "highest_r2": float(best_r2["R-squared"]),
    }
    comparison_results = save_model_comparison(
        study.best_params, X, y, X_train, X_test, y_train, y_test
    )
    best_model = comparison_results.iloc[0]
    metrics["model_comparison"] = {
        "best_model": best_model["Model"],
        "lowest_rmse": float(best_model["RMSE"]),
        "r_squared": float(best_model["R-squared"]),
    }

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, predictions, alpha=.8)
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    plt.plot(limits, limits, "r--", label="Ideal")
    plt.xlabel("Actual profit")
    plt.ylabel("Predicted profit")
    plt.title("Actual vs Predicted Profit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Refit on all available data before serializing the production model.
    final_model = make_model(study.best_params).fit(X, y)
    joblib.dump(final_model, ROOT / "startup_model.pkl.gz", compress=("gzip", 3))
    (ROOT / "evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Saved final model to {ROOT / 'startup_model.pkl.gz'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25, help="Optuna trial count")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    main(args.trials)
