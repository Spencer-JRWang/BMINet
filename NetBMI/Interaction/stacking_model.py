import warnings
warnings.filterwarnings('ignore')
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import os

class StackingModel:
    def __init__(self, output_dir='../data/predict'):
        """
        Initialize the ModelEvaluator with an output directory.

        :param output_dir: Directory to save prediction files.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def model_combinations(self):
        """
        Generate all possible combinations of base models for stacking.

        :return: List of tuples, where each tuple contains a model name and its instance.
        """
        base_model = [
            ('LGBM', LGBMClassifier(verbose=-1, n_estimators=1000, max_depth=5)),
            ('XGBoost', XGBClassifier(n_estimators=1000, max_depth=5)),
            ('CatBoost', CatBoostClassifier(verbose=False, iterations=800, max_depth=5))
        ]
        all_combinations = []
        for r in range(1, len(base_model) + 1):
            combinations_r = combinations(base_model, r)
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
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        shuffle_index = np.random.permutation(X.index)
        X = X.iloc[shuffle_index]
        y_encode = y_encode.iloc[shuffle_index]
        meta_model = LogisticRegression(max_iter=10000000)
        stacking_clf = StackingClassifier(estimators=base_model, final_estimator=meta_model, stack_method='predict_proba')
        score_st = cross_val_predict(stacking_clf, X, y_encode, cv=stratified_kfold, method="predict_proba")
        print(y_encode)
        scores_st.append(score_st[:, 1])
        scores_st = np.array(scores_st)
        scores_st = np.mean(scores_st, axis=0)
        dff = y_encode.to_frame()
        dff["IntegratedScore"] = scores_st
        return dff

    def stacking_model_search(self, df, feature_combination_dict):
        """
        Search for the best stacking model and evaluate its performance.

        :param df: DataFrame with features and target labels.
        :param feature_combination_dict: Dictionary with feature combinations for each disease pair.
        :return: List of best scores for each disease pair.
        """
        category = dict.fromkeys(df['Disease'])
        category = list(combinations(category, 2))
        Best_Scores = []
        for i in category:
            Cat_A = i[0]
            Cat_B = i[1]

            all_com = self.model_combinations()
            FPR = []
            TPR = []
            AUCs = []
            Scores = []

            df = df[df['Disease'].isin([Cat_A, Cat_B])]
            print("...Stacking model is building...")

            best = feature_combination_dict[f"{Cat_A} vs {Cat_B}"]
            for m in tqdm(all_com):
                IntegratedScore = self.stacking_model(df[best], df['Disease'].map({Cat_A: 0, Cat_B: 1}), list(m))
                Scores.append(IntegratedScore)
                fpr, tpr, thresholds = roc_curve(IntegratedScore.iloc[:, 0], IntegratedScore.iloc[:, 1])
                roc_auc = auc(fpr, tpr)
                AUCs.append(roc_auc)
                FPR.append(fpr)
                TPR.append(tpr)

            best_stacking = []
            for t in all_com[AUCs.index(max(AUCs))]:
                best_stacking.append(t[0])

            # Save the best scores to a file
            best_score_df = Scores[AUCs.index(max(AUCs))]
            Best_Scores.append(best_score_df)
            # best_score_df.to_csv(os.path.join(self.output_dir, f"{Cat_A}_{Cat_B}.txt"), sep='\t', index=False)
            print(f"Best Stacking Model detected: {best_stacking}")
            print(f"Best IntegratedScore AUC = {max(AUCs)}")

        return Best_Scores
