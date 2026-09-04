"""."""

import numpy as np
import matplotlib.pyplot as mplt


def plot_element(el, props='all', logx=True, logy=True):
    el.plot(props=props, show=False, save=True, logx=logx, logy=logx)
    # mplt.show()


def unique_sorted(*vecs):
    return np.unique(np.hstack(vecs))


def interpolate_wake(s, s_samp, W_samp):
    return np.interp(s, s_samp, W_samp, left=0.0, right=0.0)


def interpolate_impedance(w, w_samp, Z_samp):
    Z = np.interp(w, w_samp, Z_samp.imag, left=0.0, right=0.0)*1j
    Z += np.interp(w, w_samp, Z_samp.real, left=0.0, right=0.0)
    return Z


def discard_low_frequency_due_to_offset_in_wake(Z, w, min_w=6e10):
    idx = ((w < min_w) & (w > 0)).nonzero()[0]
    Z[idx] = Z[idx[-1]+1]
    idx = ((w > -min_w) & (w < 0)).nonzero()[0]
    Z[idx] = Z[idx[0]-1]
    Z[w == 0] = 1j*Z[idx[0]].imag


def plot_el(
    el, props="all", logx=False, logy=False, show=True, figname="", figsize=(8, 4)
):
    """Plot element properties, using frequency in GHz for impedances.

    Args:
        el: Element to plot.
        props (str or list, optional): Properties to plot.
        logx (bool, optional): Use logarithmic x-axis.
        logy (bool, optional): Use logarithmic y-axis.
        show (bool, optional): Show figures.
        figname (str, optional): Suffix for saved filename.
        figsize (tuple, optional): Figure size.
    """

    if props == "all":
        props = [
            "Zll", "Zdx", "Zdy", "Zqx", "Zqy",
            "Wll", "Wdx", "Wdy", "Wqx", "Wqy",
        ]
    elif isinstance(props, str):
        props = [props]

    for prop in props:

        # Let the original method create the plot
        el.plot(
            props=prop,
            logx=logx,
            logy=logy,
            show=show,
            save=False,
            figsize=figsize,
        )

        fig = mplt.gcf()
        ax = fig.axes[0]

        # Convert omega [rad/s] -> f [GHz]
        if prop.startswith("Z"):
            for line in ax.lines:
                omega = line.get_xdata()
                freq_GHz = omega / (2 * np.pi * 1e9)
                line.set_xdata(freq_GHz)

            ax.set_xlabel("Frequency [GHz]")

            # Recalculate limits after changing x-data
            ax.relim()
            ax.autoscale_view()

        fig.tight_layout()

        if show:
            fig.show()