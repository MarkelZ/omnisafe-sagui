import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("progress.csv", delimiter=",")


def plot_and_save(column: str, scale=100):
    x_label = 'TotalEnvSteps'
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_label].values, df[column].values)
    plt.scatter(df[x_label].values, df[column].values)
    plt.xlabel(x_label)
    plt.ylabel(column)
    plt.title(column)
    plt.ylim(-5, scale)
    plt.grid(True)

    fname = column.replace('/', '-')
    plt.savefig(fname + ".png")
    # plt.show()


# Plot progress
plot_and_save("Metrics/EpRet", scale=75)
plot_and_save("Metrics/EpCost", scale=50)
plot_and_save("Metrics/TestEpRet", scale=50)
plot_and_save("Metrics/TestEpCost", scale=50)

# Plot LossAlpha
# plot_and_save("PiEntropy")
