"""Plot saved DB-QITE/QCGD/GD/CGD energy trajectories."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from saveload import load_data


METHOD_STYLES = {
    "qcgd": {"color": "#20306f", "marker": "o", "label": "QCGD"},
    "cgd": {"color": "#20306f", "marker": "v", "label": "CGD"},
    "dbqite": {"color": "#6929C4", "marker": "o", "label": "DB-QITE"},
    "gd": {"color": "#6929C4", "marker": "^", "label": "GD"},
}


def plot_method_energies(ax, n_qubits: int, methods: Sequence[str]):
    """Plot selected methods' saved energies against iteration number.

    Parameters
    ----------
    n_qubits:
        Number of qubits used in the saved comparison.
    methods:
        Any selection of ``"qcgd"``, ``"dbqite"``, ``"cgd"``, and ``"gd"``.

    Returns
    -------
    tuple
        The Matplotlib ``(figure, axes)`` pair.
    """
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")
    if isinstance(methods, str):
        raise TypeError("methods must be a sequence of method-name strings.")
    if any(not isinstance(method, str) for method in methods):
        raise TypeError("Every entry in methods must be a string.")

    normalized_methods = [method.lower() for method in methods]
    if not normalized_methods:
        raise ValueError("At least one method must be specified.")

    unknown_methods = sorted(set(normalized_methods) - set(METHOD_STYLES))
    if unknown_methods:
        valid_methods = ", ".join(METHOD_STYLES)
        raise ValueError(
            f"Unknown method(s): {', '.join(unknown_methods)}. "
            f"Choose from: {valid_methods}."
        )
    if len(normalized_methods) != len(set(normalized_methods)):
        raise ValueError("Each method should be specified only once.")

    results = load_data(f"DBIvsGCI_L={n_qubits}", folder=["DBIvsGCI"])
    saved_energies = results["energies"]
    steps = np.asarray(results.get("steps", []))

    for method in normalized_methods:
        if method not in saved_energies:
            raise KeyError(f"The saved data do not contain energies for '{method}'.")

        energies = np.asarray(saved_energies[method], dtype=float)
        method_steps = steps if len(steps) == len(energies) else np.arange(len(energies))
        style = METHOD_STYLES[method]
        ax.plot(
            method_steps,
            energies,
            color=style["color"],
            marker=style["marker"],
            markersize=7,
            linewidth=1.8,
            label=style["label"],
        )

    ax.set_xlabel(r"Steps $k$")
    ax.set_ylabel(r"Energy $E_k$")
    ax.set_xticks(np.arange(len(next(iter(saved_energies.values())))))
    ax.legend()
    ax.figure.tight_layout()

__all__ = ["plot_method_energies"]
