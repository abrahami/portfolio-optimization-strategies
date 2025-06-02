from os.path import join as opj
import matplotlib.pyplot as plt
import numpy as np


def plot_nav_values(nav_values, saving_file_path, labels, save_plot=True, saving_file_name='nav_over_time.png'):
    plt.figure(figsize=(30, 15))  # Increase the figure size
    plt.plot(nav_values['Date'], nav_values['Momentum Pure TS'], linestyle='-', color='blue', label=labels[0])
    plt.plot(nav_values['Date'], nav_values['Momentum Cumulative TS'], linestyle='-', color='black',
             label=labels[1])
    plt.plot(nav_values['Date'], nav_values['Benchmark'], linestyle=':', color='red', label=labels[2])
    plt.title('Algorithms Performance Along Time', fontsize=24)
    plt.legend(loc='upper left', fontsize=18)

    # Rotate x-axis labels explicitly
    plt.xticks(rotation=45, fontsize=16)  # Ensure rotation is applied here
    plt.yticks(fontsize=20)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Net Assessment Value (NAV)', fontsize=22)

    plt.grid(True, linewidth=0.1)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.2)
    # Save the figure with high resolution
    if save_plot:
        plt.savefig(opj(saving_file_path, saving_file_name), dpi=300, bbox_inches='tight')
    plt.show()


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]
