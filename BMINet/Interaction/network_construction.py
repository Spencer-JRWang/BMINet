import numpy as np
import networkx as nx
import shap
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from itertools import combinations
from sklearn.model_selection import train_test_split

class NetworkConstructor:
    def __init__(self, cutoff, core_name="LightGBM"):
        """
        Initialize the NetworkConstructor with the dataset, feature combinations, cutoff value, and core model.
        
        :param df: DataFrame containing the data.
        :param feature_combination_dict: Dictionary with feature combinations for each pair of categories.
        :param cutoff: Threshold to determine whether an edge should be added to the network.
        :param core_name: Name of the core model to use ('LightGBM', 'XGBoost', or 'CatBoost').
        """
        self.cutoff = cutoff
        self.core_name = core_name
        self.core_model = self._select_core_model(core_name)

    def _select_core_model(self, core_name):
        """
        Select the core model based on the given core name.
        
        :param core_name: Name of the core model ('LightGBM', 'XGBoost', or 'CatBoost').
        :return: The core model.
        """
        if core_name == "LightGBM":
            # No need to return a model, as LightGBM's training code is used directly.
            return None  
        elif core_name == "XGBoost":
            return XGBClassifier(n_estimators=1000, max_depth=5)
        elif core_name == "CatBoost":
            return CatBoostClassifier(verbose=False, iterations=1500, max_depth=5)
        else:
            raise ValueError("Unsupported core. Choose from LightGBM, XGBoost, or CatBoost.")

    def construct_network(self,df,feature_combination_dict, edge_weight = False):
        """
        Construct networks for each pair of categories and return a list of interaction networks.
        
        :return: List of constructed interaction networks.
        """
        self.feature_combination_dict = feature_combination_dict
        category = dict.fromkeys(df['Disease'])
        category_pairs = list(combinations(category, 2))
        Interactions = []

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
                model = lgb.train(
                    params,
                    d_train,
                    1000,
                    valid_sets=[d_test],
                )
            else:
                model = self.core_model.fit(X, y)
            
            # Compute SHAP interaction values
            shap_interaction_values = shap.TreeExplainer(model, feature_perturbation="interventional").shap_interaction_values(X)
            interaction_matrix = np.abs(shap_interaction_values).sum(0)
            
            # Remove self-interactions
            np.fill_diagonal(interaction_matrix, 0)
            
            # Sort and select the top interactions
            inds = np.argsort(-interaction_matrix.sum(0))[:len(self.feature_combination_dict[f'{Cat_A} vs {Cat_B}'])]
            sorted_ia_matrix = interaction_matrix[inds, :][:, inds]
            feature_names = X.columns[inds]
            
            # Create a network graph
            G = nx.Graph()
            
            # Add nodes with feature names
            G.add_nodes_from(feature_names)
            
            # Add edges with weights from the interaction matrix
            for row in range(sorted_ia_matrix.shape[0]):
                for col in range(row + 1, sorted_ia_matrix.shape[1]):
                    weight = sorted_ia_matrix[row, col]
                    if edge_weight:
                        G.add_edge(feature_names[row], feature_names[col], weight=weight)
                    else:
                        if weight > self.cutoff:
                            G.add_edge(feature_names[row], feature_names[col])
            
            Interactions.append(G)

        return Interactions

    def compose_all(self, Interactions):
        """
        Combine all networks into one single network.
        
        :param Interactions: List of interaction networks.
        :return: A single merged network graph.
        """
        combined_graph = nx.compose_all(Interactions)
        return combined_graph

    def remove_isolated_nodes(self, Graph_BMI):
        """
        Remove isolated nodes from the graph.
        
        :param Graph_BMI: The network graph from which to remove isolated nodes.
        :return: The graph with isolated nodes removed.
        """
        isolated_nodes = list(nx.isolates(Graph_BMI))
        Graph_BMI.remove_nodes_from(isolated_nodes)
        return Graph_BMI
    
    def save_graph(self, Graph_BMI, path):
        """
        Save .graphml file.
        """
        nx.write_graphml(Graph_BMI, f'{path}/network.graphml')
        print(f"Network file saved to {path}/network.graphml")
        return None

