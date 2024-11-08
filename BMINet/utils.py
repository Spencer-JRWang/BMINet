import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def convert_to_number(lst):
    result = []
    for item in lst:
        if item == '':
            result.append(None)
        elif item == 'True':
            result.append(True)
        elif item == 'False':
            result.append(False)
        else:
            try:
                result.append(float(item))
            except ValueError:
                result.append(item)
    return result


def plot_multi_his():
    labels = ['A', 'B', 'C', 'D']
    # data = np.array([
    #     [65, 29, 2, 9],   # Group A
    #     [21, 70, 3, 29],   # Group B
    #     [4, 44, 8, 22],   # Group C
    #     [7, 32, 10, 83]    # Group D
    # ])

    data = np.array([
        [20, 1, 0, 0],   # Group A
        [0, 26, 0, 0],   # Group B
        [2, 5, 9, 0],   # Group C
        [0, 0, 0, 26]    # Group D
    ])

    data_ratio = data / data.sum(axis=1)[:, None]

    # colors = sns.color_palette("Pastel1", n_colors=data.shape[1])
    colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2"]

    fig, ax = plt.subplots(figsize=(5, 8))
    cumulative_data = np.zeros(len(labels))
    for i, row in enumerate(data_ratio.T):
        ax.bar(labels, row, bottom=cumulative_data, label=f'{labels[i]}', color=colors[i])
        cumulative_data += row
    ax.legend()
    ax.set_xlabel('Group')
    ax.set_ylabel('Proportion')
    
    ax.set_title('Multi-Class Prediction', fontweight='bold')
    plt.tight_layout()
    plt.savefig("./multi-class_test.pdf", format = 'pdf')
    # plt.show()


def open_test_data(data_route, with_category = True):
    f = open(data_route, "r")
    all_data = []
    all_data_new = []
    all_text = f.readlines()
    for i in all_text:
        text = i.rstrip("\n")
        text = text.split("\t")
        if with_category:
            all_data.append(text)
        else:
            all_data.append(text[1:])
    all_data = all_data[1:]
    for i in all_data:
        j = convert_to_number(i)
        all_data_new.append(j)
    f.close()
    return all_data_new



def extract_data(selected_combination, new_data):
    all_features = ['Gender_Female', 'Gender_Male', 'Age', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1', 'L1-L2_1', 'L1-L2_2',
                            'L1-L2_3', 'L1-L2_4', 'L1-L2_5', 'L1-L2_6', 'L2-L3_1', 'L2-L3_2',
                            'L2-L3_3', 'L2-L3_4', 'L2-L3_5', 'L2-L3_6', 'L3-L4_1', 'L3-L4_2',
                            'L3-L4_3', 'L3-L4_4', 'L3-L4_5', 'L3-L4_6', 'L4-L5_1', 'L4-L5_2',
                            'L4-L5_3', 'L4-L5_4', 'L4-L5_5', 'L4-L5_6', 'L5-S1_1', 'L5-S1_2',
                            'L5-S1_3', 'L5-S1_4', 'L5-S1_5', 'L5-S1_6']
    selected_indices = [all_features.index(feature) for feature in selected_combination]
    selected_data = [new_data[i] for i in selected_indices]
    return selected_data

import pandas as pd
from sklearn.model_selection import StratifiedKFold

def train_test_data(file_path,save_path):
    df_all = pd.read_csv(file_path, sep='\t')
    group_column = df_all.columns[0]
    groups = df_all[group_column]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    
    for train_index, test_index in kf.split(df_all, groups):
        train_data = df_all.iloc[train_index]
        test_data = df_all.iloc[test_index]
        train_data.to_csv(f'{save_path}/train_fold_{fold}.txt', sep='\t', index=False)
        test_data.to_csv(f'{save_path}/test_fold_{fold}.txt', sep='\t', index=False)
        
        fold += 1

import pandas as pd
from sklearn.impute import KNNImputer

def Fill_NA(df: pd.DataFrame, na_threshold: float = 0.5, knn_neighbors: int = 5, save_file: str = None) -> pd.DataFrame:
    """
    Removes columns with a missing value ratio exceeding a specified threshold, 
    and imputes remaining missing values using the KNN algorithm.

    Parameters:
    df : pd.DataFrame
        Input DataFrame with features as columns, some of which may have missing values.
    na_threshold : float, default=0.5
        The threshold proportion of missing values above which a column is removed (between 0 and 1).
    n_neighbors : int, default=5
        The number of neighboring samples used by KNNImputer for filling missing values.

    Returns:
    pd.DataFrame
        A DataFrame with columns exceeding the missing value threshold removed, 
        and remaining missing values imputed using KNN.
    """
    # Identify columns with a missing value ratio above the threshold
    na_ratio = df.isna().mean()
    cols_to_drop = na_ratio[na_ratio > na_threshold].index
    df = df.drop(columns=cols_to_drop)

    labels = df.iloc[:, 0]

    # KNN fill
    imputer = KNNImputer(n_neighbors=knn_neighbors)
    df_imputed_values = pd.DataFrame(imputer.fit_transform(df.iloc[:, 1:]), columns=df.columns[1:], index=df.index)
    df_imputed = pd.concat([labels, df_imputed_values], axis=1)

    if save_file:
        # Save the imputed DataFrame to file
        df_imputed.to_csv(save_file, index=False)
    
    return df_imputed

if __name__ == "__main__":
    plot_multi_his()