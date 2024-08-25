import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import networkx as nx
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
from sklearn.metrics import roc_curve, auc


def plot_roc_for_disease_pairs(file_path, output_dir_pdf):
    """
    Plot ROC curves for each pair of diseases with all features.

    Parameters:
    file_path (str): Path to the input data file.
    output_dir (str): Directory to save the output PDF files.

    Returns:
    None
    """
    print("...Generating feature ROC plots...")
    
    # Read the txt file
    data = pd.read_csv(file_path, delimiter='\t')
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Get unique disease categories
    diseases = data['Disease'].unique()
    
    # Generate pairs of diseases
    disease_pairs = [(diseases[i], diseases[j]) for i in range(len(diseases)) for j in range(i + 1, len(diseases))]
    
    # Plot for each disease pair
    for disease_pair in disease_pairs:
        # Create a figure and axis
        data_current = data[data["Disease"].isin(list(disease_pair))]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--')
        
        # Dictionary to store AUC values for each feature
        auc_dict = {}

        # Plot ROC curves for each feature
        for feature in data.columns[1:]:
            # Extract feature data and labels, dropping NaN values
            feature_data = data_current[[feature, 'Disease']].dropna()
            if len(feature_data) < 50:
                pass
            else:
                disease_counts = feature_data['Disease'].value_counts()

                # Find most frequent disease
                most_common_disease = disease_counts.idxmax()

                # 0-1 encoding
                feature_data['Disease'] = feature_data['Disease'].apply(lambda x: 1 if x == most_common_disease else 0)
                X = feature_data[[feature]]
                y = feature_data['Disease']

                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y, X)
                roc_auc = auc(fpr, tpr)

                # If AUC <= 0.5 then reverse the y
                if roc_auc <= 0.5:
                    y = [0 if m == 1 else 1 for m in y]
                    fpr, tpr, _ = roc_curve(y, X)
                    roc_auc = auc(fpr, tpr)

                # Plot ROC curve
                ax.plot(fpr, tpr, label=f'{feature} (AUC = {roc_auc:.4f})', lw = 1)

                # Store AUC value
                auc_dict[feature] = roc_auc
        # Sort legend labels by AUC values
        handles, labels = ax.get_legend_handles_labels()
        # print(handles)
        # print(labels)
        # print(auc_dict)
        labels_and_aucs = [(label, auc_dict[label.split()[0]]) for label in labels]
        labels_and_aucs_sorted = sorted(labels_and_aucs, key=lambda x: x[1], reverse=True)
        labels_sorted = [x[0] for x in labels_and_aucs_sorted]
        handles_sorted = [handles[labels.index(label)] for label in labels_sorted]
        print(f"=========={disease_pair[0]} vs {disease_pair[1]}==========")
        for n in labels_and_aucs:
            print(n[0])
        ax.legend(handles_sorted[:15], labels_sorted[:15], loc='lower right', fontsize=6)

        # Set title and axis labels
        plt.title(f'ROC Curve for {disease_pair[0]} vs {disease_pair[1]}')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')

        # Save as PDF file
        output_path = os.path.join(output_dir_pdf, f'{disease_pair[0]}_vs_{disease_pair[1]}_ROC.pdf')
        plt.savefig(output_path, format='pdf')
    print(f'files saved to {output_dir_pdf}')