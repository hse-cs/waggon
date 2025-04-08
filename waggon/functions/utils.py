import numpy as np
import matplotlib.pyplot as plt


def plot_colormesh_2d(func, fig, ax, x_range, y_range, name=None, nx=150, ny=150, norm=None):
    """
    Plot colormesh for 2-dimensional function.
    """
    assert func.dim == 2

    x_min, x_max = x_range
    y_min, y_max = y_range

    x_arr = np.linspace(x_min, x_max, nx)
    y_arr = np.linspace(y_min, y_max, ny)

    xs, ys = np.meshgrid(x_arr, y_arr)

    x_inp = np.zeros((ny * nx, 2))
    x_inp[:, 0] = xs.flatten()
    x_inp[:, 1] = ys.flatten()

    z_out = func(x_inp)
    z_out = z_out.squeeze(1) if z_out.ndim == 2 else z_out
    z_out = z_out.reshape((nx, ny), order='F')
    z_out = np.transpose(z_out)

    mesh = ax.pcolormesh(x_arr, y_arr, z_out, norm=norm)
    cbar = fig.colorbar(mesh, ax=ax)

    glob_min = func.glob_min.reshape(-1, 1)
    ax.scatter(func.glob_min[:, 0], func.glob_min[:, 1], color='red', s=8)

    if name is None:
        title_name = f"{func.name} (new API)" if isinstance(func, FunctionV2) else f"{func.name} (old API)"
    else:
        title_name = name

    ax.set_title(title_name)    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
