import matplotlib.pyplot as plt
import numpy as np
import datetime
import sympy as sym
import plotly.graph_objects as go


def heatmap(values, geometry):
    values = np.reshape(values, shape=geometry.shape)
    plt.imshow(values, cmap='cool', origin='lower', interpolation='None', interpolation_stage='data')
    plt.colorbar()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #plt.savefig(now + ".svg", format ='svg', bbox_inches = 'tight', pad_inches = 0)
    plt.show()


def graph_flat_walk(positions_split, geometry):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor('#f5f8fc')
    for positions in positions_split:
        positions = np.array(positions)
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

    positions = np.array([p for sub in positions_split for p in sub])

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