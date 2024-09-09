import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def convert_to_number(lst):
    result = []
    for item in lst:
        if item == '':
            result.append(None)
        else:
            try:
                result.append(float(item))
            except ValueError:
                result.append(item)
    return result

def plot_multi_his():
    labels = ['A', 'B', 'C', 'D']
    data = np.array([
        [65, 29, 2, 9],   # Group A
        [21, 70, 3, 29],   # Group B
        [4, 44, 8, 22],   # Group C
        [7, 32, 10, 83]    # Group D
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
    plt.savefig("./multi-class.pdf", format = 'pdf')
    # plt.show()
    

if __name__ == "__main__":
    plot_multi_his()