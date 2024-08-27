import os
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class StackingModel:
    def __init__(self, base_models=None, meta_model=None, cv_splits=5, random_state=42, save = False):
        """
        Initialize the StackingModel with an output directory, base models, and meta model.

        :param save: Directory to save prediction files. False means do not save files
        :param base_models: List of tuples with model names and their instances.
        :param meta_model: The final estimator model for stacking.
        :param cv_splits: Number of splits for cross-validation.
        :param random_state: Seed for random number generation.
        """

        # Default base models if not provided
        self.base_models = base_models or [
            ('LGBM', LGBMClassifier(verbose=-1, n_estimators=1000, max_depth=5)),
            ('XGBoost', XGBClassifier(n_estimators=1000, max_depth=5)),
            ('CatBoost', CatBoostClassifier(verbose=False, iterations=800, max_depth=5))
        ]
        
        # Default meta model if not provided
        self.meta_model = meta_model or LogisticRegression(max_iter=10000000)
        self.save = save
        self.cv_splits = cv_splits
        self.random_state = random_state

    def model_combinations(self):
        """
        Generate all possible combinations of base models for stacking.

        :return: List of tuples, where each tuple contains a model name and its instance.
        """
        all_combinations = []
        for r in range(1, len(self.base_models) + 1):
            combinations_r = combinations(self.base_models, r)
            all_combinations.extend(combinations_r)
        return all_combinations

    def stacking_model(self, X, y_encode, base_model):
        """
        Build and evaluate a stacking model.

        :param X: Features for the model.
        :param y_encode: Encoded target labels.
        :param base_model: List of base models for stacking.
        :return: DataFrame with the integrated score.
        """
        scores_st = []
        X = X.reset_index(drop=True)
        y_encode = y_encode.reset_index(drop=True)
        
        stratified_kfold = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=self.random_state)
        
        stacking_clf = StackingClassifier(
            estimators=base_model, 
            final_estimator=self.meta_model, 
            stack_method='predict_proba'
        )
        
        score_st = cross_val_predict(stacking_clf, X, y_encode, cv=stratified_kfold, method="predict_proba")
        scores_st.append(score_st[:, 1])
        scores_st = np.array(scores_st)
        scores_st = np.mean(scores_st, axis=0)
        
        dff = y_encode.to_frame()
        dff["IntegratedScore"] = scores_st
        return dff

    def stacking_model_search(self, df, feature_combination_dict, save_format=False):
        """
        Search for the best stacking model and evaluate its performance.

        :param df: DataFrame with features and target labels.
        :param feature_combination_dict: Dictionary with feature combinations for each disease pair.
        :param save_format: File format to save the best scores ('txt', 'csv').
        :return: List of best scores for each disease pair.
        """
        categories = list(combinations(df['Disease'].unique(), 2))
        Best_Scores = {}
        
        for Cat_A, Cat_B in categories:
            all_com = self.model_combinations()
            FPR, TPR, AUCs, Scores = [], [], [], []

            df_subset = df[df['Disease'].isin([Cat_A, Cat_B])]
            print(f"Stacking model is building for {Cat_A} vs {Cat_B}...")

            best_features = feature_combination_dict.get(f"{Cat_A} vs {Cat_B}", df.columns.drop('Disease'))
            
            for m in tqdm(all_com):
                IntegratedScore = self.stacking_model(df_subset[best_features], df_subset['Disease'].map({Cat_A: 0, Cat_B: 1}), list(m))
                Scores.append(IntegratedScore)
                fpr, tpr, _ = roc_curve(IntegratedScore.iloc[:, 0], IntegratedScore["IntegratedScore"])
                roc_auc = auc(fpr, tpr)
                AUCs.append(roc_auc)
                FPR.append(fpr)
                TPR.append(tpr)

            best_idx = AUCs.index(max(AUCs))
            best_stacking = [t[0] for t in all_com[best_idx]]
            best_score_df = Scores[best_idx]
            Best_Scores[f"{Cat_A} vs {Cat_B}"] = best_score_df
            
            if self.save:
                file_path = os.path.join(self.save, f"{Cat_A}_{Cat_B}.txt")
                best_score_df.to_csv(file_path + f'/{Cat_A} vs {Cat_B}.txt', sep = '\t', index=False)
            else:
                pass

            print(f"Best Stacking Model detected: {best_stacking}")
            # print(f"Best IntegratedScore AUC = {max(AUCs)}")

        return Best_Scores
