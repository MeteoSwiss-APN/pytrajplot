"""Generate Altitude Figure."""


# Third-party
import matplotlib.pyplot as plt
import numpy as np


def alt_fig(key, trajectory):
    print("--- generating altiude plots for " + key)

    # Some example data to display
    x = np.linspace(0, 2 * np.pi, 400)
    y = np.sin(x ** 2)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
    fig.suptitle("Aligning x-axis using sharex")
    ax1.plot(x, y)
    ax2.plot(x + 1, -y)
    ax3.plot(x, y)
    ax4.plot(x + 1, -y)

    trajectory.to_csv("src/pytrajplot/plt/" + key + ".csv", index=False)
    plt.savefig("src/pytrajplot/plt/" + key + ".png")
    return
