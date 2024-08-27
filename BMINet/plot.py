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
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import calibration_curve


def plot_single_roc(file_path, output_dir_pdf = False):
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
    dict_AUC = {}
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
        dict_AUC[f"{disease_pair}"] = labels_and_aucs
        ax.legend(handles_sorted[:15], labels_sorted[:15], loc='lower right', fontsize=6)

        # Set title and axis labels
        plt.title(f'ROC Curve for {disease_pair[0]} vs {disease_pair[1]}')
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')

        # Save as PDF file
        if output_dir_pdf:
            output_path = os.path.join(output_dir_pdf, f'{disease_pair[0]}_vs_{disease_pair[1]}_ROC.pdf')
            plt.savefig(output_path, format='pdf')
            print(f'files saved to {output_dir_pdf}')
        plt.show()
        plt.close()

    return dict_AUC
    



def plot_pca(file_path, output_dir = False):
    """
    Plot PCA for each disease group.

    Parameters:
    file_path (str): Path to the input data file.
    output_dir (str): Directory to save the output PDF files.

    Returns:
    None
    """
    print("...Generating PCA plots...")

    # Read the txt file
    data = pd.read_csv(file_path, delimiter='\t')

    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    # Extract feature columns (assuming 'Disease' is the first column)
    features = data.columns[1:]

    # Remove columns with any NaN values
    data_no_nan = data.dropna(axis=1, how='any')
    features_no_nan = data_no_nan.columns[1:]  # Recalculate features without NaNs

    if len(features_no_nan) < 2:
        raise ValueError("Not enough features without NaN values for PCA.")

    # Perform PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_no_nan[features_no_nan])

    # Create a DataFrame with the PCA results
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Disease'] = data_no_nan['Disease']

    # Plot PCA
    plt.figure(figsize=(8.5, 5))
    palette = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2"]
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Disease',
        palette=palette,
        data=pca_df,
        legend='full',
        alpha=1
    )

    # Plot ellipses
    for disease, color in zip(pca_df['Disease'].unique(), palette):
        subset = pca_df[pca_df['Disease'] == disease]
        if subset.shape[0] > 2:
            cov = np.cov(subset[['PC1', 'PC2']].values, rowvar=False)
            mean = subset[['PC1', 'PC2']].mean().values
            v, w = np.linalg.eigh(cov)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # Convert to degrees
            ell = Ellipse(mean, v[0], v[1], 180.0 + angle, color=color, alpha=0.3)
            plt.gca().add_patch(ell)

    # Set title and axis labels
    # plt.title('PCA of Diseases')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}%)')
    # Save as PDF file
    if output_dir:
        output_path = os.path.join(output_dir, 'PCA.pdf')
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_ml_roc(best_scores, output_dir=False):
    """
    Plot ROC curves for multiple groups and optionally save the plot as a PDF.

    This function generates ROC (Receiver Operating Characteristic) curves for each group in the `best_scores`
    dictionary. It computes the AUC (Area Under the Curve) for each group and overlays all ROC curves in a single plot.
    If an output directory is specified, the plot will be saved as a PDF file.

    Parameters:
    ----------
    best_scores : dict
        A dictionary where keys are group names (e.g., 'Group1 vs Group2') and values are DataFrames. Each DataFrame
        should have two columns: the first column contains true labels, and the second column contains predicted scores
        or probabilities.

    output_dir : str or bool, optional (default=False)
        The directory where the plot will be saved as a PDF. If False, the plot is not saved to disk.

    Returns:
    -------
    None
        Displays the ROC plot on the screen. If `output_dir` is specified, saves the plot as "ROC_all.pdf" in the
        specified directory.

    Example:
    -------
    best_scores = {
        "DiseaseA vs DiseaseB": pd.DataFrame([[0, 0.1], [1, 0.9], [0, 0.4], [1, 0.8]]),
        "DiseaseC vs DiseaseD": pd.DataFrame([[0, 0.2], [1, 0.85], [0, 0.35], [1, 0.7]])
    }
    plot_ml_roc(best_scores, output_dir="path/to/save/dir")
    """
    
    # Generate a colormap with a distinct color for each group
    colors = plt.cm.get_cmap('tab10', len(best_scores))
    idx = 0  # Index to track colors

    # Create a new figure for plotting
    plt.figure(figsize=(5, 5))

    # Loop through each group in best_scores
    for group in best_scores:
        data = best_scores[f"{group}"]  # Retrieve the DataFrame for the current group

        # Compute the false positive rate, true positive rate, and thresholds for the ROC curve
        fpr, tpr, thresholds = roc_curve(data.iloc[:, 0], data.iloc[:, 1])

        # Calculate the AUC (Area Under the Curve)
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve for the current group
        plt.plot(fpr, tpr, color=colors(idx), lw=2, label=f"{group} (AUC = {roc_auc:.2f})")

        # Optionally, fill the area under the curve (currently set to transparent)
        plt.fill_between(fpr, tpr, color=colors(idx), alpha=0)

        # Increment the color index
        idx += 1

    # Plot the diagonal line representing random chance
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random chance')

    # Set the limits for the x-axis (Specificity) and y-axis (Sensitivity)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Label the axes and set the plot title
    plt.xlabel('Specificity', fontsize=10)
    plt.ylabel('Sensitivity', fontsize=10)
    plt.title('ROC plot of all groups')

    # Add a grid for better readability
    plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)

    # Add a legend to the plot
    plt.legend(loc="lower right", fontsize=8)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # If an output directory is provided, save the plot as a PDF
    if output_dir:
        output_path = os.path.join(output_dir, "ROC_all.pdf")
        plt.savefig(output_path, format="pdf")
        print(f"ROC plot saved to {output_path}")

    # Display the plot
    plt.show()

    # Close the plot to free up memory
    plt.close()


def plot_score_histogram(best_scores, output_dir=False):
    """
    Plot histograms of predicted scores for each group and optionally save the plot as a PDF.

    This function generates histograms for the predicted scores in the `best_scores` dictionary.
    For each group, it creates two histograms: one for Class 0 and one for Class 1, overlaid in the same plot.
    If an output directory is specified, the plot will be saved as a PDF file.

    Parameters:
    ----------
    best_scores : dict
        A dictionary where keys are group names (e.g., 'Group1 vs Group2') and values are DataFrames. Each DataFrame
        should have two columns: the first column contains true labels (0 or 1), and the second column contains
        predicted scores or probabilities.

    output_dir : str or bool, optional (default=False)
        The directory where the plot will be saved as a PDF. If False, the plot is not saved to disk.

    Returns:
    -------
    None
        Displays the histogram plot on the screen. If `output_dir` is specified, saves the plot as "Score_Histogram.pdf"
        in the specified directory.

    Example:
    -------
    best_scores = {
        "DiseaseA vs DiseaseB": pd.DataFrame([[0, 0.1], [1, 0.9], [0, 0.4], [1, 0.8]]),
        "DiseaseC vs DiseaseD": pd.DataFrame([[0, 0.2], [1, 0.85], [0, 0.35], [1, 0.7]])
    }
    plot_score_histogram(best_scores, output_dir="path/to/save/dir")
    """
    
    # Generate a colormap with a distinct color for each group
    colors = plt.cm.get_cmap('tab10', len(best_scores))
    plt.figure(figsize=(10, 5))  # Set the figure size

    # Loop through each group in best_scores
    for idx, group in enumerate(best_scores):
        data = best_scores[group]  # Retrieve the DataFrame for the current group

        # Plot the histogram for Class 0
        plt.hist(data.iloc[:, 1][data.iloc[:, 0] == 0], bins=30, alpha=0.3, color=colors(idx),
                 label=f"{group} - Class {group.split(' vs ')[0]}")

        # Plot the histogram for Class 1
        plt.hist(data.iloc[:, 1][data.iloc[:, 0] == 1], bins=30, alpha=0.75, color=colors(idx),
                 label=f"{group} - Class {group.split(' vs ')[1]}")

    # Set the labels and title
    plt.xlabel('Predicted Score')
    plt.ylabel('Count')
    plt.title('Histogram of Predicted Scores with Categories')

    # Add a legend to the plot
    plt.legend(loc=[0.6, 0.5], fontsize=8)

    # Add a grid for better readability
    plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # If an output directory is provided, save the plot as a PDF
    if output_dir:
        output_path = os.path.join(output_dir, "Score_Histogram.pdf")
        plt.savefig(output_path, format="pdf")
        print(f"Histogram plot saved to {output_path}")

    # Display the plot
    plt.show()

    # Close the plot to free up memory
    plt.close()


def plot_precision_recall(best_scores, output_dir=False):
    """
    Plot Precision-Recall curves for each group in best_scores and optionally save the plot as a PDF.

    This function generates Precision-Recall curves for the predicted scores in the `best_scores` dictionary.
    For each group, it plots the precision against recall values. If an output directory is specified,
    the plot will be saved as a PDF file.

    Parameters:
    ----------
    best_scores : dict
        A dictionary where keys are group names (e.g., 'Group1 vs Group2') and values are DataFrames. Each DataFrame
        should have two columns: the first column contains true labels (0 or 1), and the second column contains
        predicted scores or probabilities.

    output_dir : str or bool, optional (default=False)
        The directory where the plot will be saved as a PDF. If False, the plot is not saved to disk.

    Returns:
    -------
    None
        Displays the Precision-Recall plot on the screen. If `output_dir` is specified, saves the plot as 
        "Precision_Recall_all.pdf" in the specified directory.

    Example:
    -------
    best_scores = {
        "DiseaseA vs DiseaseB": pd.DataFrame([[0, 0.1], [1, 0.9], [0, 0.4], [1, 0.8]]),
        "DiseaseC vs DiseaseD": pd.DataFrame([[0, 0.2], [1, 0.85], [0, 0.35], [1, 0.7]])
    }
    plot_precision_recall(best_scores, output_dir="path/to/save/dir")
    """
    
    # Generate a colormap with a distinct color for each group
    colors = plt.cm.get_cmap('tab10', len(best_scores))
    plt.figure(figsize=(5, 5))  # Set the figure size

    # Loop through each group in best_scores
    for idx, group in enumerate(best_scores):
        data = best_scores[group]  # Retrieve the DataFrame for the current group

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(data.iloc[:, 0], data.iloc[:, 1])

        # Plot the precision-recall curve
        plt.plot(recall, precision, color=colors(idx), lw=2, label=f"{group}")

    # Set the labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall plot of all groups')

    # Add a legend to the plot
    plt.legend(loc="lower left", fontsize=8)

    # Add a grid for better readability
    plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # If an output directory is provided, save the plot as a PDF
    if output_dir:
        output_path = os.path.join(output_dir, "Precision_Recall_all.pdf")
        plt.savefig(output_path, format="pdf")
        print(f"Precision-Recall plot saved to {output_path}")

    # Display the plot
    plt.show()

    # Close the plot to free up memory
    plt.close()



def plot_calibration_curve(best_scores, output_dir=False):
    """
    Plot calibration curves for each group in best_scores and optionally save the plot as a PDF.

    This function generates calibration curves for the predicted probabilities in the `best_scores` dictionary.
    For each group, it plots the mean predicted probability against the fraction of positives.
    If an output directory is specified, the plot will be saved as a PDF file.

    Parameters:
    ----------
    best_scores : dict
        A dictionary where keys are group names (e.g., 'Group1 vs Group2') and values are DataFrames.
        Each DataFrame should have two columns: the first column contains true labels (0 or 1),
        and the second column contains predicted probabilities.

    output_dir : str or bool, optional (default=False)
        The directory where the plot will be saved as a PDF. If False, the plot is not saved to disk.

    Returns:
    -------
    None
        Displays the calibration plot on the screen. If `output_dir` is specified, saves the plot as 
        "Calibration_all.pdf" in the specified directory.

    Example:
    -------
    best_scores = {
        "DiseaseA vs DiseaseB": pd.DataFrame([[0, 0.1], [1, 0.9], [0, 0.4], [1, 0.8]]),
        "DiseaseC vs DiseaseD": pd.DataFrame([[0, 0.2], [1, 0.85], [0, 0.35], [1, 0.7]])
    }
    plot_calibration_curve(best_scores, output_dir="path/to/save/dir")
    """
    
    # Generate a colormap with a distinct color for each group
    colors = plt.cm.get_cmap('tab10', len(best_scores))
    plt.figure(figsize=(10, 5))  # Set the figure size

    # Loop through each group in best_scores
    for idx, group in enumerate(best_scores):
        data = best_scores[group]  # Retrieve the DataFrame for the current group

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(data.iloc[:, 0], data.iloc[:, 1], n_bins=10)

        # Plot the calibration curve
        plt.plot(prob_pred, prob_true, marker='o', color=colors(idx), label=f"{group}", linewidth=2)

    # Plot the reference line for perfect calibration
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)

    # Set the labels and title
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration plot of all groups')

    # Add a legend to the plot
    plt.legend(loc="upper left", fontsize=8)

    # Add a grid for better readability
    plt.grid(linestyle="--", color="gray", linewidth=0.5, zorder=0, alpha=0.5)

    # Adjust the layout to prevent overlap
    plt.tight_layout()

    # If an output directory is provided, save the plot as a PDF
    if output_dir:
        output_path = os.path.join(output_dir, "Calibration_all.pdf")
        plt.savefig(output_path, format="pdf")
        print(f"Calibration plot saved to {output_path}")

    # Display the plot
    plt.show()

    # Close the plot to free up memory
    plt.close()
