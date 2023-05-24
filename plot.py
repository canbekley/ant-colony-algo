from functools import reduce
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import PIL
import io

from models import ConstructionCycle


# TODO: cleanup this module and properly document it.


def plot_construction_graph(filename: str, cc: ConstructionCycle, colored_pheromones: bool = False):
    data, layout = _get_base_graph_layout(cc, colored_pheromones)
    fig = go.Figure(data=data, layout=layout)
    fig.update_traces(line=dict(width=4))
    if colored_pheromones:
        fig.update_traces(
            marker=dict(
                colorscale="BuPu",
                colorbar=dict(
                    title="Pheromone",
                    len=0.5,
                    thickness=20,
                    tickvals=[0, 2, 4, 6, 8, 10],
                    ticktext=[0, 0.4, 0.8, 1.2, 1.6, 2],
                ),
            )
        )
    fig.write_image(file=filename, format="png")


def plot_construction_solution_path(filename: str, cc: ConstructionCycle, colored_pheromones: bool = False):
    data, layout = _get_base_graph_layout(cc, colored_pheromones)
    sp_data = _get_solution_path_data(cc)
    fig = go.Figure(data=data + sp_data, layout=layout)
    fig.update_traces(line=dict(width=4))
    fig.write_image(file=filename, format="png")


def plot_construction_animation(filename: str, cc: ConstructionCycle):
    graph_data, layout = _get_base_graph_layout(cc)
    animation_data, frames = _get_construction_animation(cc)
    data = reduce(lambda x, y: x + y, [animation_data, graph_data])
    fig = go.Figure(data=data, layout=layout, frames=frames)
    _write_to_gif(filename, fig, graph_data)


def _get_base_graph_layout(cc: ConstructionCycle, colored_pheromones: bool = False):
    vertices = pd.DataFrame(
        columns=["x", "y", "info"],
        data=[(v[0], v[1], v.info if v.info else "vertex") for v in cc.construction_graph.vertices],
    )

    edges = pd.DataFrame(
        columns=["edge", "v1_x", "v1_y", "v2_x", "v2_y", "pheromone"],
        data=[(str(e), e.i[0], e.i[1], e.j[0], e.j[1], e.pheromone) for e in cc.construction_graph.edges],
    )

    fig_1 = px.scatter(
        vertices,
        x="x",
        y="y",
        color="info",
        color_discrete_map={
            "vertex": "rgba(0, 0, 0, 0.1)",
            "destination": "rgba(245, 72, 66, 0.8)",
            "origin": "rgba(28, 176, 72, 0.8)",
        },
        range_x=[0.9, 6.1],
        range_y=[0.9, 6.1],
        template="simple_white",
        height=700,
        width=800,
    )
    fig_1.update_traces(marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")), selector=dict(mode="markers"))

    fig_2 = [
        px.line(
            x=[row["v1_x"], row["v2_x"]],
            y=[row["v1_y"], row["v2_y"]],
            color_discrete_sequence=(
                px.colors.sample_colorscale("BuPu", np.clip(np.array([row["pheromone"]]) - 0.5, 0, 1))
                if colored_pheromones
                else ["rgba(0, 0, 0, 0.1)"]
            ),
        ).data
        for _, row in edges.iterrows()
    ]

    fig_1.update_layout(
        showlegend=True, xaxis={"visible": False}, yaxis={"visible": False}, legend={"font": {"size": 20}}
    )

    fig_data = reduce(lambda x, y: x + y, [*fig_2, fig_1.data])
    return fig_data, fig_1.layout


def _get_solution_path_data(cc: ConstructionCycle):
    name_generator = _get_ant_names()
    data = _flatten_list(
        [
            [(name, v[0] + j[0], v[1] + j[1]) for v in ant.solution_path.convert_to_list_of_vertices()]
            for ant, (name, j) in zip(cc.ants, [(next(name_generator), _jitter()) for _ in range(len(cc.ants))])
        ]
    )
    solution_paths = pd.DataFrame(columns=["ant", "x", "y"], data=data)
    fig = px.line(data_frame=solution_paths, x="x", y="y", color="ant")
    return fig.data


def _get_construction_animation(cc: ConstructionCycle):
    name_generator = _get_ant_names()
    sp = [ant.solution_path.convert_to_list_of_vertices() for ant in cc.ants]
    max_len = max([len(s) for s in sp])
    sp = [
        [
            (n, e, v[0] + j[0], v[1] + j[1])
            for n, (e, v), j in zip([next(name_generator)] * len(s), enumerate(s), [_jitter()] * len(s))
        ]
        for s in sp
    ]
    data = _flatten_list(_flatten_list([[(a[0], i, a[2], a[3]) for i in range(a[1], max_len)] for a in s]) for s in sp)
    df = pd.DataFrame(columns=["ant", "construction_step", "x", "y"], data=data)

    fig = px.line(
        df,
        x="x",
        y="y",
        animation_frame="construction_step",
        animation_group="ant",
        # line_dash_sequence=["dash"],
        color="ant",
        hover_name="ant",
    )
    return fig.data, fig.frames


def _write_to_gif(filename, fig, graph_data):
    # generate images for each step in animation
    frames = []
    for s, fr in enumerate(fig.frames):
        # set main traces to appropriate traces within plotly frame
        fig.update(data=fr.data + graph_data)
        # generate image of current state
        frames.append(PIL.Image.open(io.BytesIO(fig.to_image(format="png"))))

    # create animated GIF
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=1000,
        loop=0,
    )


def _get_ant_names():
    while True:
        yield "Armin"
        yield "Berthold"
        yield "Reiner"
        yield "Levi"
        yield "Eren"
        yield "Erwin"
        yield "Annie"


def _jitter():
    return (np.random.rand(1, 2)[0] - 0.5) * 0.15


def _flatten_list(x):
    return reduce(lambda a, b: a + b, x)
