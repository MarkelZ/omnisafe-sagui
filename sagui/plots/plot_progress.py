import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def subplot(df, algo, x_label, y_label):
    x = df[x_label].values
    y = df[y_label].values

    plt.plot(x, y, label=algo)
    plt.scatter(x, y, label=algo)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(y_label)
    plt.grid(True)
    plt.legend()


# Get the current directory as a Path object
current_directory = Path.cwd()

# Specify the pattern for CSV files
csv_pattern = '*.csv'

# Use the glob method of the Path object to find all files matching the pattern
csv_files = current_directory.glob(csv_pattern)

# Display the list of CSV files
for csv_file in csv_files:
    fname = csv_file.name
    algo = fname[9:-4]
    df = pd.read_csv(fname, delimiter=",")

    plt.figure(figsize=(15, 5))

    x_label = 'TotalEnvSteps'

    y_label = 'Metrics/EpRet'
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, subplot 1
    subplot(df, algo, x_label, y_label)

    y_label = 'Metrics/EpCost'
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, subplot 2
    subplot(df, algo, x_label, y_label)

    plt.savefig(algo + '.png')
