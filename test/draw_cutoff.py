import numpy as np
import matplotlib.pyplot as plt


def c2_envelope(x, p):
    return (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * x**p
        + p * (p + 2.0) * x**(p + 1)
        - (p * (p + 1.0) / 2.0) * x**(p + 2)
    )


def c3_envelope(x, p):
    a = -((p + 1.0) * (p + 2.0) * (p + 3.0) / 6.0)
    b = (p * (p + 2.0) * (p + 3.0) / 2.0)
    c = -(p * (p + 1.0) * (p + 3.0) / 2.0)
    d = (p * (p + 1.0) * (p + 2.0) / 6.0)

    return (
        1.0
        + a * x**p
        + b * x**(p + 1)
        + c * x**(p + 2)
        + d * x**(p + 3)
    )


def plot_cutoffs(p_list, num_points=500):
    """
    p_list: list of integers, e.g. [3,4,5,6]
    """
    x = np.linspace(0.0, 1.0, num_points)

    plt.figure()

    for p in p_list:
        y_c2 = c2_envelope(x, p)
        y_c3 = c3_envelope(x, p)

        plt.plot(x, y_c2, label=f"C2 p={p}")
        plt.plot(x, y_c3, linestyle="--", label=f"C3 p={p}")

    plt.xlabel("r / cutoff")
    plt.ylabel("envelope")
    plt.title("C2 vs C3 Polynomial Cutoff")
    plt.legend()
    plt.grid(True)
    plt.savefig('cutoff.pdf')


if __name__ == "__main__":
    p_list = [5, 6, 7]

    plot_cutoffs(p_list)
