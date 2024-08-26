import shap
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from itertools import combinations
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import os

class SHAPVisualizer:
    def __init__(self, df_all, output_dir='../figure/CT/Exp', feature_combinations=None):
        """
        Initialize the SHAP Visualizer.

        :param df_all: DataFrame with all data.
        :param output_dir: Directory to save plots.
        :param feature_combinations: Dictionary with feature combinations for each disease pair.
        """
        self.df_all = df_all
        self.output_dir = output_dir
        self.feature_combinations = feature_combinations or {}
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'Dependence'), exist_ok=True)

    def get_model(self, core_name):
        """
        Get the model based on the core_name.

        :param core_name: Name of the core model to use.
        :return: A model instance.
        """
        if core_name == "LightGBM":
            return None
        elif core_name == "XGBoost":
            return XGBClassifier(n_estimators=1000, max_depth=5)
        elif core_name == "CatBoost":
            return CatBoostClassifier(verbose=False, iterations=1500, max_depth=5)
        else:
            raise ValueError("Unsupported core. Choose from LightGBM, XGBoost, or CatBoost.")

    def train_model(self, X, y, model_name):
        """
        Train the model and return the trained model and SHAP values.

        :param X: Features.
        :param y: Target variable.
        :param model_name: Name of the core model to use.
        :return: Trained model and SHAP values.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_name == "LightGBM":
            d_train = lgb.Dataset(X_train, label=y_train)
            d_test = lgb.Dataset(X_test, label=y_test)
            params = {
                "max_bin": 512,
                "learning_rate": 0.05,
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 10,
                "verbose": -1,
                "boost_from_average": True,
                "early_stopping_rounds": 50,
                "verbose_eval": 1000
            }
            model = lgb.train(
                params,
                d_train,
                1000,
                valid_sets=[d_test]
            )
        else:
            model = self.get_model(model_name)
            model.fit(X_train, y_train)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        return model, shap_values

    def plot_shap(self, shap_values, category, name_prefix):
        """
        Plot and save SHAP summary plots and dependence plots.

        :param shap_values: SHAP values.
        :param category: Tuple of disease categories.
        :param name_prefix: Prefix for filenames.
        """
        # Save summary plot
        fig = shap.plots.beeswarm(shap_values, show=False, alpha=0.7)
        plt.title(f"{category[0]} vs {category[1]}", fontweight='bold', fontsize=10)
        plt.xlabel("Impact on model output")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{name_prefix}.pdf", bbox_inches='tight')
        plt.close()

        # Save heatmap plot
        fig = shap.plots.heatmap(shap_values, show=False)
        plt.title(f"{category[0]} vs {category[1]}", fontweight='bold', fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{name_prefix}_heat.pdf", bbox_inches='tight')
        plt.close()

    def plot_dependence(self, shap_values, X, category, name_prefix):
        """
        Plot and save SHAP dependence plots for each feature.

        :param shap_values: SHAP values.
        :param X: Features.
        :param category: Tuple of disease categories.
        :param name_prefix: Prefix for filenames.
        """
        start_color = (1, 0, 0)  # red
        middle_color = (1, 0.843, 0)  # gold
        end_color = (0, 0.392, 0)  # dark green
        cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, middle_color, end_color], N=1000)

        for m in tqdm(X.columns.to_list()):
            fig, ax = plt.subplots(tight_layout=True, figsize=(6,4))
            plt.title(f"{category[0]} vs {category[1]} feature {m} dependence", fontweight='bold', fontsize=10)
            ax.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
            shap.plots.scatter(shap_values[:, f"{m}"], color=shap_values, cmap=cmap, ax=ax, show=False)
            plt.savefig(f"{self.output_dir}/Dependence/{name_prefix}_{m}.pdf", bbox_inches='tight')
            plt.close()

    def visualize(self):
        """
        Main method to perform training, SHAP value computation, and plotting for each category combination.
        """
        categories = ["A", "B", "C", "D"]
        category_pairs = list(combinations(categories, 2))

        for category in category_pairs:
            Cat_A, Cat_B = category
            df = self.df_all[self.df_all['Disease'].isin([Cat_A, Cat_B])]
            X = df.drop("Disease", axis=1)
            feature_comb = self.feature_combinations.get(f"{Cat_A} vs {Cat_B}", X.columns)
            X = X[feature_comb]
            y = df['Disease'].map({Cat_A: 0, Cat_B: 1})

            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            shuffle_index = np.random.permutation(X.index)
            X = X.iloc[shuffle_index]
            y = y.iloc[shuffle_index]

            for model_name in ["LightGBM", "XGBoost", "CatBoost"]:
                trained_model, shap_values = self.train_model(X, y, model_name)
                name_prefix = f"{Cat_A}_vs_{Cat_B}_{model_name}"

                # Plot SHAP results
                self.plot_shap(shap_values, category, name_prefix)
                self.plot_dependence(shap_values, X, category, name_prefix)
