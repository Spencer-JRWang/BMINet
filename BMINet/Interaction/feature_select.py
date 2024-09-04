import warnings
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from tqdm import tqdm  # Progress bar for loops
from itertools import combinations  # To generate combinations of disease categories

class FeatureSelector:
    def __init__(self, core_name="LightGBM"):
        """
        Initialize the FeatureSelector with a core machine learning model.
        
        Parameters:
        core_name (str): The name of the core model to use for feature selection. 
                         Options are "LightGBM", "XGBoost", or "CatBoost".
        """
        self.core = self._get_core(core_name)  # Initialize the core model based on user input
        self.category_combinations = None  # Placeholder for storing disease category combinations

    def _get_core(self, core_name):
        """
        Private method to initialize the core model based on the given name.
        
        Parameters:
        core_name (str): The name of the core model to initialize.
        
        Returns:
        model: An instance of the specified core model.
        
        Raises:
        ValueError: If the core_name is not one of the supported options.
        """
        if core_name == "LightGBM": 
            return LGBMClassifier(n_estimators=1000, max_depth=5, verbose=-1)
        elif core_name == "XGBoost":
            return XGBClassifier(n_estimators=1000, max_depth=5)
        elif core_name == "CatBoost":
            return CatBoostClassifier(verbose=False, iterations=800, max_depth=5)
        else:
            raise ValueError("Unsupported core. Choose from LightGBM, XGBoost, or CatBoost.")

    def select(self, df):
        """
        Perform feature selection for each pair of disease categories in the dataset.
        
        Parameters:
        df (DataFrame): The input DataFrame containing features and a 'Disease' column.
        
        Returns:
        dict: A dictionary where keys are pairs of disease categories, and values 
              are lists of the best features for distinguishing between those categories.
        """
        # Generate all possible combinations of disease categories
        self.category_combinations = list(combinations(dict.fromkeys(df['Disease']), 2))
        dict_groups_features = {}  # Dictionary to store best features for each category pair

        print(f"==================== Current Core: {self.core.__class__.__name__} ====================")
        # Loop through each pair of disease categories
        for cat_A, cat_B in self.category_combinations:
            print(f"Searching For Group: {cat_A} vs {cat_B}")
            # Filter the DataFrame to include only the current category pair
            df_filtered = df[df['Disease'].isin([cat_A, cat_B])]
            # Perform Recursive Feature Elimination (RFE) and get the best features
            best_features = self._model_rfe(df_filtered, cat_A, cat_B)
            # Store the best features for the current category pair
            dict_groups_features[f'{cat_A} vs {cat_B}'] = best_features

        return dict_groups_features

    def _model_rfe(self, df, cat_A, cat_B):
        """
        Private method to perform Recursive Feature Elimination (RFE) and 
        select the best features for distinguishing between two disease categories.
        
        Parameters:
        df (DataFrame): The input DataFrame filtered to include only the two disease categories.
        cat_A (str): The name of the first disease category.
        cat_B (str): The name of the second disease category.
        
        Returns:
        list: The list of best features for distinguishing between the two categories.
        """
        # Separate features (X) and target variable (y)
        X = df.drop("Disease", axis=1).reset_index(drop=True)
        y = df['Disease'].map({cat_A: 0, cat_B: 1}).reset_index(drop=True)

        # Shuffle the data to prevent ordering effects
        shuffle_index = np.random.permutation(X.index)
        X = X.iloc[shuffle_index]
        y = y.iloc[shuffle_index]

        outcome_feature = []  # List to store selected features for each iteration
        outcome_score = []  # List to store cross-validation scores for each iteration

        # Perform RFE for different numbers of selected features
        for i in tqdm(range(1, X.shape[1] + 1)):
            rfe = RFE(self.core, n_features_to_select=i)  # Initialize RFE with i features
            rfe.fit(X, y)  # Fit RFE model
            selected_features = X.columns[rfe.support_]  # Get the selected features
            # Perform cross-validation to evaluate the selected features
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(self.core, X[selected_features], y, cv=cv)
            outcome_feature.append(selected_features)  # Store the selected features
            outcome_score.append(scores.mean())  # Store the mean cross-validation score

        # Identify the best feature combination with the highest validation score
        max_predict_data = max(outcome_score)
        best_features = list(outcome_feature[outcome_score.index(max_predict_data)])
        print(f"Best Features Combination Detected: {best_features}")
        print(f"Best Validation Score: {max_predict_data}")

        return best_features
