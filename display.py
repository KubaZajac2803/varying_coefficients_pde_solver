import matplotlib.pyplot as plt
import numpy as np
import datetime
import sympy as sym
import plotly.graph_objects as go


def heatmap(values, geometry, num_walks):
    values = np.reshape(values, shape=geometry.shape)
    plt.imshow(values, cmap='cool', origin='lower', interpolation='None', interpolation_stage='data')
    plt.colorbar()
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    plt.savefig(f'plots/{now}_shape_{geometry.shape}_walks_{num_walks}.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


def graph_flat_walk(positions_split, geometry):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor('#f5f8fc')
    if len(positions_split) > 1:
        for positions in positions_split:
            positions = np.array(positions)
            plt.plot(positions[:, 0], positions[:, 1], linewidth=0.3, c='black', zorder=1)
    else:
        positions = positions_split[0]
        plt.plot(positions[:, 0], positions[:, 1], linewidth=0.3, c='black', zorder=1)
    plt.scatter(positions_split[0][0][0], positions_split[0][0][1], s=40, c='orange', marker='X', zorder=2)
    plt.scatter(positions_split[-1][-1][0], positions_split[-1][-1][1], s=40, c='orange', marker='X', zorder=2)
    plt.xlim(0, geometry.bdr_max_x)
    plt.ylim(0, geometry.bdr_max_y)
    plt.xlabel("u")
    plt.ylabel("v")
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #plt.savefig(now + ".svg", format ='svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()

def graph_walk_on_surface(positions_split, geometry, surface_parameterization):
    u, v = sym.var('u v')
    phi = sym.lambdify((u, v), sym.Matrix(surface_parameterization), 'numpy')

    if len(positions_split) > 1:
        positions = np.array([p for sub in positions_split for p in sub])
    else:
        positions = positions_split[0]
    u_param = np.linspace(0, geometry.bdr_max_x, 100)
    v_param = np.linspace(0, geometry.bdr_max_y, 100)
    U, V = np.meshgrid(u_param, v_param)

    X, Y, Z = map(np.asarray, phi(U, V))
    X = np.asarray(X).squeeze()
    Y = np.asarray(Y).squeeze()
    Z = np.asarray(Z).squeeze()

    u_pos, v_pos = positions[:, 0], positions[:, 1]
    x_pos, y_pos, z_pos = phi(u_pos, v_pos)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        opacity=0.6,
        showscale=False,
    ))

    fig.add_trace(go.Scatter3d(
        x=x_pos.flatten(),
        y=y_pos.flatten(),
        z=z_pos.flatten(),
        mode='lines',
        line=dict(width=6, color='red'),
    ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
    )
    fig.show()

def heatmap_riemannian(values, geometry, surface_parameterization, num_walks):
    u, v = sym.var('u v')
    phi = sym.lambdify((u, v), sym.Matrix(surface_parameterization), 'numpy')

    u_param = np.linspace(0, geometry.bdr_max_x, 100)
    v_param = np.linspace(0, geometry.bdr_max_y, 100)
    U, V = np.meshgrid(u_param, v_param)

    X, Y, Z = map(np.asarray, phi(U, V))
    X = np.asarray(X).squeeze()
    Y = np.asarray(Y).squeeze()
    Z = np.asarray(Z).squeeze()

    pts = np.array(geometry.points_to_check())
    u_pos, v_pos = pts[:, 0], pts[:, 1]

    xyz = np.array(phi(u_pos, v_pos))
    x_pos = xyz[0].reshape(-1)
    y_pos = xyz[1].reshape(-1)
    z_pos = xyz[2].reshape(-1)

    cool_plotly = [
        [0.0, "rgb(0, 255, 255)"],  # cyan
        [1.0, "rgb(255, 0, 255)"],  # magenta
    ]

    fig = go.Figure(
        go.Scatter3d(
            x=x_pos,
            y=y_pos,
            z=z_pos,
            mode='markers',
            marker=dict(
                size=5,
                symbol='circle',
                color=values,
                colorscale=cool_plotly,
                showscale = True,
            )
        )
    )

    """fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        opacity=0.4,
        showscale=False,
    ))"""

    fig.update_layout(
        scene=dict(aspectmode='data'),
    )
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    fig.write_html(f'{now}_{geometry.sample_num}_samples_{num_walks}_walks.html')
    fig.show()

def heatmap_riemannian_conform(values, geometry):
    grid_size = 30
    u = np.linspace(-1, 1, grid_size)
    v = np.linspace(-1, 1, grid_size)
    U, V = np.meshgrid(u, v)
    mask = np.square(U) + np.square(V) <= geometry.radius
    xyz = geometry.to_3D(np.array([U, V]))#*mask[None, :, :]
    pts = geometry.to_3D(geometry.points_to_check().swapaxes(0, 1))
    x_dot = pts[0, :]
    y_dot = pts[1, :]
    z_dot = pts[2, :]

    x_pos = xyz[0, :]
    y_pos = xyz[1, :]
    z_pos = xyz[2, :]

    cool_plotly = [
        [0.0, "rgb(0, 255, 255)"],  # cyan
        [1.0, "rgb(255, 0, 255)"],  # magenta
    ]

    fig = go.Figure(
             go.Scatter3d(
                 x=x_dot,
                 y=y_dot,
                 z=z_dot,
                 mode='markers',
                 marker=dict(
                     size=5,
                     symbol='circle',
                     color=values,
                     colorscale=cool_plotly,
                     showscale=True
                 )
             )
    )

    #fig.add_trace(go.Surface(
    #    x=x_pos, y=y_pos, z=z_pos,
    #    opacity=0.4,
    #    showscale=False,
    #))

    fig.update_layout(
        scene=dict(aspectmode='data'),
    )
    
    fig.write_html("bdr_zero_constant_source.html")
