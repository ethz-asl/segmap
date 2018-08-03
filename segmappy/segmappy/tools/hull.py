import numpy as np


def point_in_hull(point, hull, tolerance=1e-12):
    return all((np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)


def n_points_in_hull(points, hull):
    n_points = 0
    for i in range(points.shape[0]):
        if point_in_hull(points[i, :], hull):
            n_points = n_points + 1
    return n_points


def are_in_hull(points, hull):
    ins = []
    outs = []
    for i in range(points.shape[0]):
        if point_in_hull(points[i, :], hull):
            ins.append(i)
        else:
            outs.append(i)
    return ins, outs
