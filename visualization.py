"""Figure helpers for the ising_ed_pec_ieee notebook.

All plotting/styling lives here; the notebook cells pass data in and get a figure out.
The chip maps follow the style of Fig. 30 of arXiv:2607.25998.
"""

from math import comb

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Circle, Rectangle, Wedge

# one typography scheme for every figure in the notebook
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


def x_labels(layout, n_data):
    """Per-observable tick labels carrying the physical qubit indices (site i = qubit i)."""
    return [f"$X_{{{layout[i]}}}$" for i in range(n_data)]


def plot_layout(backend, layout, n_data):
    """Chip cartoon of the embedding: data qubits green, check qubits orange."""
    from qiskit.visualization import plot_coupling_map

    return plot_coupling_map(
        num_qubits=backend.num_qubits,
        qubit_coordinates=None,
        coupling_map=list(backend.coupling_map.get_edges()),
        figsize=(9, 9),
        qubit_color=[
            "#4CAF50" if q in layout[:n_data] else "#FF9800" if q in layout[n_data:] else "#DDDDDD"
            for q in range(backend.num_qubits)
        ],
        qubit_size=220,
        line_width=2,
        font_size=90,
    )


def plot_exact(obs_exact, tick_labels, title):
    plt.figure(figsize=(12, 4))
    plt.plot(obs_exact, "o-")
    plt.title(title)
    plt.xticks(np.arange(len(obs_exact)), tick_labels)
    plt.xlabel("Observable")
    plt.ylabel(r"$\langle X \rangle$")
    plt.grid()


def draw_toy_circuit(generate_ed_ising, zz_coeff, x_coeff, include_checks=True):
    """The boxing pipeline on a 3-plaquette-ring miniature (1 Trotter step) so the box
    structure is legible; every box carries the Twirl / InjectNoise annotations, and with
    ``include_checks`` the terminal xslow non-Markovian error check pattern is appended as in the
    production pipeline."""
    import networkx as nx
    from qiskit.circuit import ClassicalRegister
    from qiskit_mitigation.postselection import XSlowGate
    from samplomatic.transpiler import generate_boxing_pass_manager

    toy, _, _ = generate_ed_ising(nx.cycle_graph(3), 1, zz_coeff, x_coeff)
    toy.add_register(ClassicalRegister(3, "data"), ClassicalRegister(3, "check"))
    toy.barrier()
    toy.measure(range(3), range(3))
    toy.measure(range(3, 6), range(3, 6))
    toy_boxed = generate_boxing_pass_manager(
        enable_gates=True,
        enable_measures=True,
        inject_noise_targets="gates",
        inject_noise_strategy="individual_modification",
        inject_noise_site="after",
        twirling_strategy="active_circuit",
        measure_annotations="all",
    ).run(toy)
    if include_checks:
        toy_boxed.add_register(ClassicalRegister(3, "data_ps"), ClassicalRegister(3, "check_ps"))
        toy_boxed.barrier()
        for qb in range(6):
            toy_boxed.append(XSlowGate(), [qb])
        toy_boxed.measure(range(3), toy_boxed.cregs[2])
        toy_boxed.measure(range(3, 6), toy_boxed.cregs[3])
    return toy_boxed.draw("mpl", fold=-1, scale=0.6)


# --- chip-level noise maps ---------------------------------------------------------

BANDS = [1e-5, 2e-5, 3e-5, 4e-5, 6e-5, 1e-4, 2e-4, 3e-4, 4e-4, 6e-4, 1e-3]
CMAP = plt.get_cmap("YlOrBr")
NORM = BoundaryNorm(BANDS, CMAP.N, extend="both")


def _color(rate):
    return "white" if rate < BANDS[0] else CMAP(NORM(min(rate, BANDS[-1] * 0.999)))


def _layer_sparse(mit, weights):
    """Per-layer labelled terms from the saved run, box-local -> physical qubits."""
    phys = np.sort(mit["layout"])
    return [
        [
            (p, tuple(int(phys[q]) for q in qs if q >= 0), r)
            for p, qs, r in zip(mit[f"label_paulis_{i}"], mit[f"label_qubits_{i}"], w, strict=True)
        ]
        for i, w in enumerate(weights)
    ]


def _aggregate(layers):
    """w1[qubit][P] and w2[(a,b)][PaPb]: rates summed over the 3 layers."""
    w1, w2 = {}, {}
    for terms in layers:
        for p, qs, r in terms:
            if len(qs) == 1:
                w1.setdefault(qs[0], dict.fromkeys("XYZ", 0.0))[p] += r
            else:
                (a, pa), (b, pb) = sorted(zip(qs, p, strict=True))
                w2.setdefault((a, b), {x + y: 0.0 for x in "XYZ" for y in "XYZ"})[pa + pb] += r
    return w1, w2


def draw_noise_map(mit, backend, reduced=False):
    """Chip map of the learned model: X/Y/Z wheel per qubit, 3x3 two-qubit Pauli grid per
    coupler, log-banded colors, dashed outlines for unused hardware.

    With ``reduced=True``, keeps only the error terms the checks cannot see: detectability
    depends on circuit position, so each layer's 0/1 site scales are averaged over its uses.
    """
    from qiskit_ibm_runtime.visualization.embeddings import Embedding

    if reduced:
        scales = [mit["site_scales"][mit["site_layer"] == i].mean(axis=0) for i in range(3)]
        weights = [mit[f"rates_{i}"] * scales[i] for i in range(3)]
        title = f"Reduced noise model ($\\gamma$ = {mit['gammas'][1]:.1f})"
    else:
        weights = [mit[f"rates_{i}"] for i in range(3)]
        title = f"Full noise model ($\\gamma$ = {mit['gammas'][0]:.0f})"
    w1, w2 = _aggregate(_layer_sparse(mit, weights))

    xy = np.array([(c, -r) for r, c in Embedding.from_backend(backend).coordinates])
    fig, ax = plt.subplots(figsize=(13, 7))
    for a, b in {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}:
        if (a, b) in w2:  # 3x3 Pauli grid laid along the bond (columns: qubit a, rows: qubit b)
            d = xy[b] - xy[a]
            u = d / np.hypot(*d)
            v = np.array([-u[1], u[0]])
            cl, cw = (np.hypot(*d) - 0.6) / 3, 0.17
            for i, Pa in enumerate("XYZ"):
                for j, Pb in enumerate("XYZ"):
                    ax.add_patch(
                        Rectangle(
                            xy[a] + u * (0.3 + i * cl) + v * ((j - 1.5) * cw),
                            cl,
                            cw,
                            angle=np.degrees(np.arctan2(u[1], u[0])),
                            facecolor=_color(w2[a, b][Pa + Pb]),
                            edgecolor="black",
                            lw=0.4,
                            zorder=2,
                        )
                    )
        else:
            ax.plot(
                *zip(xy[a], xy[b], strict=True), ls="--", lw=0.7, color="black", alpha=0.5, zorder=1
            )
    for q, (xq, yq) in enumerate(xy):
        if q in w1:  # three-sector wheel: X top, Y lower left, Z lower right
            for P, t0 in (("X", 30), ("Y", 150), ("Z", 270)):
                ax.add_patch(
                    Wedge(
                        (xq, yq),
                        0.3,
                        t0,
                        t0 + 120,
                        facecolor=_color(w1[q][P]),
                        edgecolor="black",
                        lw=0.6,
                        zorder=3,
                    )
                )
            ax.annotate(
                str(q),
                (xq + 0.39, yq - 0.39),
                fontsize=9,
                color="gray",
                ha="left",
                va="top",
                zorder=4,
            )  # southeast, clear of the bond grids
        else:
            ax.add_patch(
                Circle(
                    (xq, yq),
                    0.24,
                    facecolor="none",
                    edgecolor="black",
                    ls="--",
                    lw=0.7,
                    alpha=0.5,
                    zorder=3,
                )
            )
    ux = xy[sorted(w1)]
    ax.set_xlim(ux[:, 0].min() - 2.2, ux[:, 0].max() + 2.2)
    ax.set_ylim(ux[:, 1].min() - 1.6, ux[:, 1].max() + 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=15)
    cb = fig.colorbar(
        cm.ScalarMappable(norm=NORM, cmap=CMAP),
        ax=ax,
        fraction=0.035,
        pad=0.02,
        extend="both",
        ticks=BANDS,
    )
    cb.set_label("coefficient (log bands; white < 1e-05)", fontsize=11)
    cb.ax.set_yticklabels([f"{b:.0e}".replace("e-0", "e-") for b in BANDS], fontsize=10)
    _noise_map_legends(fig, ax)


def _noise_map_legends(fig, ax):
    """Weight-1 wheel and weight-2 grid keys, in a reserved band left of the lattice."""
    fig.subplots_adjust(left=0.17)
    axl = ax.inset_axes([-0.185, 0.70, 0.13, 0.24])
    for P, t0 in (("X", 30), ("Y", 150), ("Z", 270)):
        axl.add_patch(
            Wedge((0.5, 0.45), 0.38, t0, t0 + 120, facecolor="white", edgecolor="black", lw=0.8)
        )
        axl.annotate(
            P,
            (0.5 + 0.2 * np.cos(np.radians(t0 + 60)), 0.45 + 0.2 * np.sin(np.radians(t0 + 60))),
            ha="center",
            va="center",
            fontsize=9,
        )
    axl.set_title("weight-1", fontsize=10)
    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.set_aspect("equal")
    axl.axis("off")
    axm = ax.inset_axes([-0.185, 0.32, 0.14, 0.30])
    for i, Pa in enumerate("XYZ"):
        for j, Pb in enumerate("XYZ"):
            axm.add_patch(
                Rectangle(
                    (i / 3, 1 - (j + 1) / 3),
                    1 / 3,
                    1 / 3,
                    facecolor="white",
                    edgecolor="black",
                    lw=0.6,
                )
            )
            axm.annotate(
                Pa + Pb, ((i + 0.5) / 3, 1 - (j + 0.5) / 3), ha="center", va="center", fontsize=7.5
            )
        axm.annotate(Pa, ((i + 0.5) / 3, 1.08), ha="center", fontsize=8.5)
        axm.annotate("XYZ"[i], (-0.13, 1 - (i + 0.5) / 3), ha="center", va="center", fontsize=8.5)
    axm.annotate("qubit a", (0.5, 1.27), ha="center", fontsize=9)
    axm.annotate("qubit b", (-0.33, 0.5), rotation=90, va="center", fontsize=9)
    axm.set_xlim(-0.35, 1.05)
    axm.set_ylim(-0.05, 1.35)
    axm.set_aspect("equal")
    axm.axis("off")


# --- run diagnostics ---------------------------------------------------------------


def plot_trex(trex_rescale, tick_labels):
    _fig, axt = plt.subplots(figsize=(12, 3))
    axt.stem(np.arange(len(trex_rescale)), (trex_rescale - 1) * 100)
    axt.set_xticks(np.arange(len(trex_rescale)), tick_labels)
    axt.set_xlabel("Observable")
    axt.set_ylabel("Readout correction (%)")
    axt.set_title("TREX rescale factors")
    plt.tight_layout()


def plot_postselection(mit):
    """Accepted shots per PEC randomization against a single binomial at the mean
    acceptance rate: agreement means acceptance is independent of the sampled circuit
    instance, the condition under which pooling accepted shots across randomizations
    is a consistent estimator."""
    _fig, ax = plt.subplots(figsize=(7.5, 3.8))
    counts = mit["acc_counts_post"]
    K, p = 64, counts.mean() / 64
    ks = np.arange(K + 1)
    pmf = np.array([comb(K, k) * p**k * (1 - p) ** (K - k) for k in ks])
    ax.hist(
        counts,
        bins=np.arange(-0.5, K + 1.5),
        density=True,
        alpha=0.6,
        color="#da1e28",
        label="measured",
    )
    ax.plot(ks, pmf, "k-", lw=1.5, label=f"Binomial(64, {p:.3f})")
    ax.set_xlim(-0.5, max(int(counts.max()) + 3, 20))
    ax.set_xlabel("Accepted shots per randomization")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()


def plot_convergence(mit, obs_exact, n_data):
    """Running site-averaged estimate vs randomizations for both PEC arms (the S5-consistent
    signed-ratio estimator, evaluated on growing prefixes of the sweep)."""

    def running(prefix):
        bits = np.squeeze(
            np.unpackbits(mit[f"data_{prefix}"], axis=-1)[..., :n_data] ^ mit[f"flips_{prefix}"]
        )
        mask = np.squeeze(mit[f"mask_{prefix}"])
        signs = 1 - 2 * (np.squeeze(mit[f"signs_{prefix}"]).sum(axis=-1) % 2)
        qv = (1 - 2 * bits.astype(int)) * mit["trex_rescale"]
        u = (signs[:, None, None] * mask[..., None] * qv).sum(axis=1).mean(axis=1)
        v = signs * mask.sum(axis=1)
        Rs = np.arange(500, len(u) + 1, 500)
        est, err = [], []
        for R in Rs:
            e = u[:R].sum() / v[:R].sum()
            est.append(e)
            err.append(np.sqrt(((u[:R] - e * v[:R]) ** 2).sum()) / abs(v[:R].sum()))
        return Rs, np.array(est), np.array(err)

    _fig, ax = plt.subplots(figsize=(12, 5))
    ideal_avg = float(np.mean(obs_exact))
    ax.axhline(ideal_avg, color="black", label="ideal")
    ax.fill_between(
        [-400, 24000],
        ideal_avg - 0.025,
        ideal_avg + 0.025,
        color="grey",
        alpha=0.22,
        label=r"$\pm 0.025$",
    )
    for prefix, label, color in (
        ("van", "vanilla PEC", "#8a3ffc"),
        ("post", "PEC + error detection", "#da1e28"),
    ):
        Rs, est, err = running(prefix)
        ax.errorbar(
            Rs,
            est,
            yerr=err,
            marker="o",
            linestyle="",
            markerfacecolor="none",
            color=color,
            alpha=0.85,
            capsize=3,
            label=label,
        )
    ax.set_xlim(-400, 24000)
    ax.set_ylim(ideal_avg - 0.08, ideal_avg + 0.08)
    ax.set_xlabel("# randomizations")
    ax.set_ylabel(r"Site-averaged $\langle X \rangle$")
    ax.legend(ncols=2)


def plot_final(obs_exact, baseline, ed, pec, post, gammas, tick_labels, title):
    """Per-site <X> for every method, with an rms-deviation inset.
    baseline/ed/pec/post are (values, errors) pairs; gammas is (gamma, gamma_post)."""
    x = np.arange(len(obs_exact))
    _fig, ax = plt.subplots(figsize=(13, 5))
    h_ideal = ax.errorbar(x, obs_exact, fmt="-", capsize=4, label="ideal", color="black")
    h_base = ax.errorbar(
        x, baseline[0], yerr=baseline[1], fmt=".--", capsize=4, label="baseline", color="#0f62fe"
    )
    h_ed = ax.errorbar(
        x,
        ed[0],
        yerr=ed[1],
        fmt="^",
        capsize=4,
        label="error detection",
        color="#009d9a",
        alpha=0.8,
    )
    h_pec = ax.errorbar(
        x,
        pec[0],
        yerr=pec[1],
        fmt="x",
        capsize=4,
        label=f"PEC ($\\gamma$={gammas[0]:.0f})",
        color="#8a3ffc",
        alpha=0.7,
    )
    h_post = ax.errorbar(
        x,
        post[0],
        yerr=post[1],
        fmt="d",
        capsize=4,
        label=f"PEC + error detection ($\\gamma$={gammas[1]:.0f})",
        color="#da1e28",
    )
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("Observable")
    ax.set_ylabel("Expectation value")
    # Zoom the y-axis onto the data: the ideal, baseline, error-detection, and PEC + error
    # detection curves (with error bars) set the range, so an unconverged PEC arm may run off
    # scale instead of blowing out the axis. The upper margin leaves room for the inset.
    series = [np.asarray(obs_exact)] + [
        np.asarray(v) + s * np.asarray(e) for v, e in (baseline, ed, post) for s in (-1, 1)
    ]
    lo, hi = min(a.min() for a in series), max(a.max() for a in series)
    span = max(hi - lo, 0.1)
    ax.set_ylim(lo - 0.1 * span, hi + 1.05 * span)
    # legend ordered to match the curves' vertical positions in the chart
    ax.legend(
        handles=[h_ideal, h_pec, h_post, h_ed, h_base],
        ncols=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )
    ax.set_title(title, pad=44)

    # inset: rows bottom-to-top so it reads top-to-bottom: PEC, PEC+ED, ED, baseline
    axi = ax.inset_axes([0.36, 0.68, 0.28, 0.26])
    methods = [
        ("baseline", baseline[0], "#0f62fe"),
        ("QED", ed[0], "#009d9a"),
        ("PEC+QED", post[0], "#da1e28"),
        ("PEC", pec[0], "#8a3ffc"),
    ]
    for k, (_nm, vals, color) in enumerate(methods):
        axi.barh(k, np.sqrt(np.mean((vals - np.array(obs_exact)) ** 2)), color=color, alpha=0.9)
    axi.set_yticks(range(4), [m[0] for m in methods], fontsize=10)
    axi.set_title("RMS deviation from ideal", fontsize=11)
    axi.tick_params(labelsize=10)
    axi.patch.set_alpha(1.0)
    axi.set_zorder(5)
