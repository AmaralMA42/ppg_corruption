import matplotlib.pyplot as plt

def plot_all_trajectories(estrat_t, estrat_medio):
    plt.figure()

    for i in range(estrat_t.shape[1]):
        plt.plot(estrat_t[0, i, :], alpha=0.2)
        plt.plot(estrat_t[1, i, :], alpha=0.2)
        plt.plot(estrat_t[2, i, :], alpha=0.2)

    plt.plot(estrat_medio[0], linewidth=2, label="C")
    plt.plot(estrat_medio[1], linewidth=2, label="D")
    plt.plot(estrat_medio[2], linewidth=2, label="P")

    plt.ylim(0, 1)
    plt.legend()
    plt.show()