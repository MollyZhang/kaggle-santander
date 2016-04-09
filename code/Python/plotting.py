from matplotlib import pyplot as plt


def plot_cutting_off(df):
    plt.plot(df['threshhold'], df['train auc'])
    plt.plot(df['threshhold'], df['validation auc'])
    plt.xlabel('threshhold')
    plt.ylabel('auc score')
    plt.legend(['training', 'validation'])
    plt.title('determining cutoff threshhold and auc score for linear model')
    plt.show()
