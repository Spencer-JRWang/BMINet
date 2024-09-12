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


def open_test_data(data_route):
    f = open(data_route, "r")
    all_data = []
    all_data_new = []
    all_text = f.readlines()
    for i in all_text:
        text = i.rstrip("\n")
        text = text.split("\t")
        text = text[1:]
        all_data.append(text)
    all_data = all_data[1:]
    for i in all_data:
        j = convert_to_number(i)
        all_data_new.append(j)
    f.close()
    return all_data_new

if __name__ == "__main__":
    plot_multi_his()