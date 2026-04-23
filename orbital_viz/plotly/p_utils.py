import numpy as np
import plotly.graph_objects as go

_MPL_TO_PLOTLY_DASH = {
    "-": "solid",
    "--": "dash",
    "-.": "dashdot",
    ":": "dot",
}


def resolve_color(color):
    """
    Convert Matplotlib default color cycle (C0, C1, ...) to Plotly-compatible colors.
    """
    mpl_cycle = [
        "#1f77b4",  # C0
        "#ff7f0e",  # C1
        "#2ca02c",  # C2
        "#d62728",  # C3
        "#9467bd",  # C4
        "#8c564b",  # C5
        "#e377c2",  # C6
        "#7f7f7f",  # C7
        "#bcbd22",  # C8
        "#17becf",  # C9
    ]

    if isinstance(color, str) and color.startswith("C"):
        try:
            idx = int(color[1])
            return mpl_cycle[idx]
        except:
            return color

    return color


def plot_vector(
    fig,
    origin,
    vec,
    color="red",
    scale=1.0,
    alpha=1.0,
    linewidth=6,
    min_length=1.0,
    label=None,
    linestyle="-",
    ref_length=None,
    cone_fraction=0.03,
):
    """
    Plot a 3D vector using line + cone, with cone size scaled
    from a reference orbital length instead of the vector magnitude.
    """
    origin = np.asarray(origin, dtype=float)
    vec = np.asarray(vec, dtype=float)

    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return None

    color = resolve_color(color)

    v_hat = vec / norm
    length = max(scale * norm, min_length)
    tip = origin + length * v_hat

    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], tip[0]],
            y=[origin[1], tip[1]],
            z=[origin[2], tip[2]],
            mode="lines",
            line=dict(color=color, width=linewidth),
            opacity=alpha,
            showlegend=False,
            hoverinfo="skip",
        )
    )

    if ref_length is None:
        ref_length = length

    cone_size = cone_fraction * ref_length

    fig.add_trace(
        go.Cone(
            x=[tip[0]],
            y=[tip[1]],
            z=[tip[2]],
            u=[v_hat[0]],
            v=[v_hat[1]],
            w=[v_hat[2]],
            sizemode="absolute",
            sizeref=cone_size,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            opacity=alpha,
            anchor="tip",
        )
    )

    if label is not None:
        add_legend_entry(
            fig,
            label=label,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )

    return {"origin": origin, "tip": tip}


def add_legend_entry(
    fig,
    label,
    color="royalblue",
    linewidth=2,
    linestyle="-",
):
    """
    Add a single legend entry to a Plotly figure without plotting real data.

    Prevents duplicate labels automatically.
    """

    # --- Initialize storage if not present ---
    if not hasattr(fig, "_legend_labels"):
        fig._legend_labels = set()

    # --- Avoid duplicates ---
    if label in fig._legend_labels:
        return

    fig._legend_labels.add(label)

    dash = _MPL_TO_PLOTLY_DASH.get(linestyle, "solid")

    # --- Dummy trace (NaN so it does not appear in plot) ---
    trace = go.Scatter3d(
        x=[None],
        y=[None],
        z=[None],
        mode="lines",
        line=dict(
            color=color,
            width=linewidth,
            dash=dash,
        ),
        name=label,
        showlegend=True,
    )

    fig.add_trace(trace)


def show_figure(fig):
    fig.update_layout(
        scene=dict(
            xaxis_title="x [km]",
            yaxis_title="y [km]",
            zaxis_title="z [km]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.show()
