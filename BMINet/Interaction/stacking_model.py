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
import pickle

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
        self.df = df
        categories = list(combinations(df['Disease'].unique(), 2))
        Best_Scores = {}
        Best_Model_Combination = {}
        
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

            Best_Model_Combination[f"{Cat_A} vs {Cat_B}"] = list(all_com[AUCs.index(max(AUCs))])
            
            if self.save:
                file_path = os.path.join(self.save, f"{Cat_A}_{Cat_B}.txt")
                best_score_df.to_csv(file_path + f'/{Cat_A} vs {Cat_B}.txt', sep = '\t', index=False)
            else:
                pass

            print(f"Best Stacking Model detected: {best_stacking}")
            # print(f"Best IntegratedScore AUC = {max(AUCs)}")

        self.Best_Model_Combinations = Best_Model_Combination

        return Best_Scores

    def single_predict(self, group, new_data, use_our_model = False):
        """
        Conduct a single individual prediction on the given new data.

        :param group: The group you want to predict, for example, "A vs B".
                    This should be a string indicating the comparison between two classes.
        :param new_data: A list of feature values for the new data you want to predict.
                        The order of values must correspond to the feature combination used for the specified group.
        :param use_our_model: True or False, you can choose whether to use our trained model.
        :return: The predicted probability of belonging to each class (Cat_A, Cat_B).
                Returns an array where each element is the probability that the input data belongs to a specific class.
        :raises ValueError: If the group format is incorrect, if new_data has incorrect length, or if features are missing.
        :raises KeyError: If the group is not found in Best_Model_Combinations or feature_combination_dict.
        :raises RuntimeError: If the model fails to fit or predict due to unexpected issues.
        """

        # Validate the group format
        if not isinstance(group, str) or " vs " not in group:
            raise ValueError("The group format must be a string in the format 'A vs B'.")

        # Ensure the group exists in Best_Model_Combinations
        if group not in self.Best_Model_Combinations:
            raise KeyError(f"The group '{group}' was not found in Best_Model_Combinations.")

        # Extract the best model combination for the specified group
        current_model_combination = self.Best_Model_Combinations[group]

        # Create a Stacking classifier using the best model combination and a meta-model
        try:
            stacking_clf = StackingClassifier(
                estimators=current_model_combination,
                final_estimator=self.meta_model,
                stack_method='predict_proba'
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create StackingClassifier: {e}")

        # Extract the class names from the group string
        try:
            Cat_A, Cat_B = group.split(" vs ")
        except ValueError:
            raise ValueError("The group format is incorrect; it must be 'A vs B'.")

        # Ensure the feature combination exists for the specified group
        if f'{Cat_A} vs {Cat_B}' not in self.feature_combination_dict:
            raise KeyError(f"Feature combination for '{Cat_A} vs {Cat_B}' not found in feature_combination_dict.")

        # Filter the dataset to include only the rows corresponding to the specified classes
        df_filtered = self.df[self.df['Disease'].isin([Cat_A, Cat_B])]
        if df_filtered.empty:
            raise ValueError(f"No data found for the classes '{Cat_A}' and '{Cat_B}'.")

        # Select the features used for this group and the target labels
        X = df_filtered.drop("Disease", axis=1)
        X = X[self.feature_combination_dict[f'{Cat_A} vs {Cat_B}']]
        y = df_filtered['Disease']

        # Shuffle the data and reset indices for randomness
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        shuffle_index = np.random.permutation(X.index)
        X = X.iloc[shuffle_index]
        y = y.iloc[shuffle_index]

        # Map the target labels to binary values (0 for Cat_A, 1 for Cat_B)
        y = y.map({Cat_A: 0, Cat_B: 1})

        # Train the Stacking classifier on the training data
        try:
            if not use_our_model:
                print("Stacking Model Training...")
                stacking_clf.fit(X, y)
            else:
                print("Loading Stacking Model...")
                current_dir = os.path.dirname(__file__)
                pkl_path = os.path.join(current_dir, f'model_{Cat_A} vs {Cat_B}.pkl')
                with open(pkl_path, 'rb') as file:
                    stacking_clf = pickle.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to fit StackingClassifier: {e}")

        # Validate the new_data input
        feature_names = self.feature_combination_dict[f'{Cat_A} vs {Cat_B}']
        if not isinstance(new_data, list):
            raise ValueError("new_data must be a list of feature values.")
        if len(new_data) != len(feature_names):
            raise ValueError(f"new_data must have {len(feature_names)} values, corresponding to the required features.")

        # Convert the new_data list into a DataFrame with the appropriate feature names
        try:
            new_data_df = pd.DataFrame([new_data], columns=feature_names)
        except Exception as e:
            raise RuntimeError(f"Failed to create DataFrame from new_data: {e}")

        # Predict the probabilities for the new data
        try:
            prediction = stacking_clf.predict_proba(new_data_df)
        except Exception as e:
            raise RuntimeError(f"Failed to predict with StackingClassifier: {e}")

        # Return the predicted probabilities
        return prediction

    def multiple_predict(self, group, new_data_list, use_our_model = False):
        """
        Conduct multiple individual predictions on a list of new data samples.

        :param group: The group you want to predict, for example, "A vs B".
                    This should be a string indicating the comparison between two classes.
        :param new_data_list: A list of lists, where each inner list contains feature values for one sample.
                            The order of values in each inner list must correspond to the feature combination used for the specified group.
        :param use_our_model: True or False, you can choose whether to use our trained model.
        :return: A list of predicted probabilities for each sample. 
                Each element in the list is an array where each element is the probability that the input data belongs to a specific class.
        :raises ValueError: If the group format is incorrect, if any new_data sample has an incorrect length, or if features are missing.
        :raises KeyError: If the group is not found in Best_Model_Combinations or feature_combination_dict.
        :raises RuntimeError: If the model fails to fit or predict due to unexpected issues.
        """

        # Validate the group format
        if not isinstance(group, str) or " vs " not in group:
            raise ValueError("The group format must be a string in the format 'A vs B'.")

        # Ensure the group exists in Best_Model_Combinations
        if group not in self.Best_Model_Combinations:
            raise KeyError(f"The group '{group}' was not found in Best_Model_Combinations.")

        # Extract the best model combination for the specified group
        current_model_combination = self.Best_Model_Combinations[group]

        # Create a Stacking classifier using the best model combination and a meta-model
        try:
            stacking_clf = StackingClassifier(
                estimators=current_model_combination,
                final_estimator=self.meta_model,
                stack_method='predict_proba'
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create StackingClassifier: {e}")

        # Extract the class names from the group string
        try:
            Cat_A, Cat_B = group.split(" vs ")
        except ValueError:
            raise ValueError("The group format is incorrect; it must be 'A vs B'.")

        # Ensure the feature combination exists for the specified group
        if f'{Cat_A} vs {Cat_B}' not in self.feature_combination_dict:
            raise KeyError(f"Feature combination for '{Cat_A} vs {Cat_B}' not found in feature_combination_dict.")

        # Filter the dataset to include only the rows corresponding to the specified classes
        df_filtered = self.df[self.df['Disease'].isin([Cat_A, Cat_B])]
        if df_filtered.empty:
            raise ValueError(f"No data found for the classes '{Cat_A}' and '{Cat_B}'.")

        # Select the features used for this group and the target labels
        X = df_filtered.drop("Disease", axis=1)
        X = X[self.feature_combination_dict[f'{Cat_A} vs {Cat_B}']]
        y = df_filtered['Disease']

        # Shuffle the data and reset indices for randomness
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        shuffle_index = np.random.permutation(X.index)
        X = X.iloc[shuffle_index]
        y = y.iloc[shuffle_index]

        # Map the target labels to binary values (0 for Cat_A, 1 for Cat_B)
        y = y.map({Cat_A: 0, Cat_B: 1})

        # Train the Stacking classifier on the training data
        try:
            if not use_our_model:
                print("Stacking Model Training...")
                stacking_clf.fit(X, y)
            else:
                print("Loading Stacking Model...")
                current_dir = os.path.dirname(__file__)
                pkl_path = os.path.join(current_dir, f'model_{Cat_A} vs {Cat_B}.pkl')
                with open(pkl_path, 'rb') as file:
                    stacking_clf = pickle.load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to fit or load StackingClassifier: {e}")

        # Validate the new_data_list input
        if not isinstance(new_data_list, list) or not all(isinstance(sample, list) for sample in new_data_list):
            raise ValueError("new_data_list must be a list of lists, where each inner list contains feature values.")

        # Check each sample in new_data_list for correct length
        feature_names = self.feature_combination_dict[f'{Cat_A} vs {Cat_B}']
        for i, new_data in enumerate(new_data_list):
            if len(new_data) != len(feature_names):
                raise ValueError(f"Sample {i} in new_data_list must have {len(feature_names)} values, corresponding to the required features.")

        # Convert the new_data_list into a DataFrame with the appropriate feature names
        try:
            new_data_df = pd.DataFrame(new_data_list, columns=feature_names)
        except Exception as e:
            raise RuntimeError(f"Failed to create DataFrame from new_data_list: {e}")

        # Predict the probabilities for each sample in new_data_list
        try:
            predictions = stacking_clf.predict_proba(new_data_df)
        except Exception as e:
            raise RuntimeError(f"Failed to predict with StackingClassifier: {e}")

        # Convert predictions to a list of arrays and return it
        return [pred for pred in predictions]
