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
import os

class SHAPVisualizer:
    def __init__(self, core_name):
        """
        Initialize the SHAP Visualizer with a core model.

        :param core_name: Name of the core model to use ('LightGBM', 'XGBoost', or 'CatBoost').
        """
        self.core_name = core_name
        self.model = self.get_model()
        self.shap_values = {}
        self.X = None
        self.y = None

    def get_model(self):
        """
        Get the model based on the core_name.

        :return: A model instance.
        """
        if self.core_name == "LightGBM":
            return lgb.LGBMClassifier(n_estimators=1000, max_depth=5)
        elif self.core_name == "XGBoost":
            return XGBClassifier(n_estimators=1000, max_depth=5)
        elif self.core_name == "CatBoost":
            return CatBoostClassifier(verbose=False, iterations=1500, max_depth=5)
        else:
            raise ValueError("Unsupported core. Choose from LightGBM, XGBoost, or CatBoost.")

    def train_model(self, df, feature_combination_dict):
        """
        Train the model and compute SHAP values.

        :param df: DataFrame with all data.
        :param selected_features: List of features to use.
        """
        self.feature_combination_dict = feature_combination_dict
        print(f"==================== Current Core: {self.core_name} ====================")
        print("Training and Explaning...")
        category = dict.fromkeys(df['Disease'])
        category_pairs = list(combinations(category, 2))
        for i, (Cat_A, Cat_B) in enumerate(category_pairs):
            df_filtered = df[df['Disease'].isin([Cat_A, Cat_B])]
            X = df_filtered.drop("Disease", axis=1)
            X = X[feature_combination_dict[f'{Cat_A} vs {Cat_B}']]
            y = df_filtered['Disease']
            
            # Shuffle and reset indices
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            shuffle_index = np.random.permutation(X.index)
            X = X.iloc[shuffle_index]
            y = y.iloc[shuffle_index]
            y = y.map({Cat_A: 0, Cat_B: 1})
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model based on the selected core
            if self.core_name == "LightGBM":
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
                self.model = lgb.train(
                    params,
                    d_train,
                    1000,
                    valid_sets=[d_test],
                )
            else:
                self.model = self.model.fit(X, y)

            explainer = shap.Explainer(self.model, X)
            curr = 1
            max_attempts = 5
            while curr <= max_attempts:
                try:
                    self.shap_values[f"{Cat_A} vs {Cat_B}"] = explainer(X)
                    break
                except Exception as e:
                    # print(f"Attempt {curr} failed: {str(e)}")
                    curr += 1
                    if curr > max_attempts:
                        print(f"Failed after {max_attempts} attempts.")

            self.X = X
            self.y = y

    def plot_shap(self, output_dir):
        """
        Plot and save SHAP summary and heatmap plots.

        :param output_dir: Directory to save the plots.
        """
        print("Saving shap plots...")
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call train_model first.")
        
        os.makedirs(output_dir, exist_ok=True)
        for i in self.shap_values:
            # Save summary plot
            fig = shap.plots.beeswarm(self.shap_values[i], show=False, alpha=0.7)
            plt.title(f"SHAP Summary Plot for {self.core_name} Group {i}", fontweight='bold', fontsize=10)
            plt.xlabel("Impact on model output")
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{i}_shap_summary_plot.pdf")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()

            # Save heatmap plot
            fig = shap.plots.heatmap(self.shap_values[i], show=False)
            plt.title(f"SHAP Heatmap Plot for {self.core_name} Group {i}", fontweight='bold', fontsize=10)
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{i}_shap_heatmap_plot.pdf")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        print(f"Shap plots saved to {output_dir}")

    def plot_dependence(self, output_dir):
        """
        Plot and save SHAP dependence plots for each feature.

        :param output_dir: Directory to save the plots.
        """
        if self.shap_values is None or self.X is None:
            raise ValueError("SHAP values or features not available. Call train_model first.")
        print("Saving dependence plots...")
        os.makedirs(output_dir, exist_ok=True)

        start_color = (1, 0, 0)  # red
        middle_color = (1, 0.843, 0)  # gold
        end_color = (0, 0.392, 0)  # dark green
        cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, middle_color, end_color], N=1000)
        for i in self.shap_values:
            current_shap_value = self.shap_values[i]
            for feature in tqdm(self.feature_combination_dict[i]):
                fig, ax = plt.subplots(tight_layout=True, figsize=(6, 4))
                plt.title(f"{feature} Dependence Plot for {self.core_name}", fontweight='bold', fontsize=10)
                ax.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)
                shap.plots.scatter(current_shap_value[:, feature], color=current_shap_value, cmap=cmap, ax=ax, show=False)
                output_path = os.path.join(output_dir, f"{self.core_name}_{feature}_dependence_plot.pdf")
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
        print(f"Dependence plots saved to {output_dir}")