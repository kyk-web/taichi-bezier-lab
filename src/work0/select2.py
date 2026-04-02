import math
import taichi as ti

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 600
BG_COLOR = 0xFFFFFF
POINT_COLOR = 0xFF3030
POLYGON_COLOR = 0x808080
CURVE_COLOR = 0x00AA33
TEXT_COLOR = 0x111111

MAX_CTRL_POINTS = 256

control_points = []
use_bspline = False


def bezier_point(points, t):
    work = [[p[0], p[1]] for p in points]
    n = len(work)
    for r in range(1, n):
        for i in range(n - r):
            work[i][0] = (1.0 - t) * work[i][0] + t * work[i + 1][0]
            work[i][1] = (1.0 - t) * work[i][1] + t * work[i + 1][1]
    return work[0][0], work[0][1]


def build_bezier_polyline():
    n = len(control_points)
    if n < 2:
        return []

    approx_len = 0.0
    for i in range(n - 1):
        x1, y1 = control_points[i]
        x2, y2 = control_points[i + 1]
        approx_len += math.hypot(x2 - x1, y2 - y1)

    samples = int(max(300, min(5000, approx_len * 1.5 + 300)))
    curve = []
    for i in range(samples):
        t = i / (samples - 1) if samples > 1 else 0.0
        curve.append(bezier_point(control_points, t))
    return curve


def build_open_uniform_knot_vector(n_ctrl, degree):
    # 开放均匀结点向量
    # 例如 degree=3, n_ctrl=6
    # knot = [0,0,0,0,1,2,3,3,3,3]
    m = n_ctrl + degree + 1
    knot = [0.0] * m

    for i in range(m):
        if i <= degree:
            knot[i] = 0.0
        elif i >= n_ctrl:
            knot[i] = float(n_ctrl - degree)
        else:
            knot[i] = float(i - degree)

    return knot


def cox_de_boor(i, k, u, knot):
    if k == 0:
        # 最后一个参数点单独补上右端闭区间
        if (knot[i] <= u < knot[i + 1]) or (
            u == knot[-1] and knot[i] <= u <= knot[i + 1]
        ):
            return 1.0
        return 0.0

    left = 0.0
    right = 0.0

    denom1 = knot[i + k] - knot[i]
    if denom1 > 1e-8:
        left = (u - knot[i]) / denom1 * cox_de_boor(i, k - 1, u, knot)

    denom2 = knot[i + k + 1] - knot[i + 1]
    if denom2 > 1e-8:
        right = (knot[i + k + 1] - u) / denom2 * cox_de_boor(i + 1, k - 1, u, knot)

    return left + right


def bspline_point(points, degree, u, knot):
    x = 0.0
    y = 0.0
    n_ctrl = len(points)
    for i in range(n_ctrl):
        b = cox_de_boor(i, degree, u, knot)
        x += points[i][0] * b
        y += points[i][1] * b
    return x, y


def build_bspline_polyline():
    n = len(control_points)
    degree = 3

    if n < 4:
        return []

    knot = build_open_uniform_knot_vector(n, degree)
    u_start = knot[degree]
    u_end = knot[n]

    approx_len = 0.0
    for i in range(n - 1):
        x1, y1 = control_points[i]
        x2, y2 = control_points[i + 1]
        approx_len += math.hypot(x2 - x1, y2 - y1)

    samples = int(max(400, min(6000, approx_len * 1.8 + 400)))

    curve = []
    for i in range(samples):
        t = i / (samples - 1) if samples > 1 else 0.0
        u = u_start + t * (u_end - u_start)
        curve.append(bspline_point(control_points, degree, u, knot))

    return curve


def draw_polyline(gui, points, color, radius):
    if len(points) < 2:
        return
    for i in range(len(points) - 1):
        gui.line(begin=points[i], end=points[i + 1], color=color, radius=radius)


def draw_control_points(gui):
    for p in control_points:
        gui.circle(pos=p, radius=5.0, color=POINT_COLOR)


def main():
    global use_bspline

    gui = ti.GUI("Select 2 - Bezier / Cubic B-spline", res=(WIDTH, HEIGHT))
    prev_left = False

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in ["c", "C"]:
                control_points.clear()
            elif gui.event.key in ["b", "B"]:
                use_bspline = not use_bspline

        left_now = gui.is_pressed(ti.GUI.LMB)
        if left_now and not prev_left:
            if len(control_points) < MAX_CTRL_POINTS:
                mx, my = gui.get_cursor_pos()
                control_points.append((mx, my))
        prev_left = left_now

        gui.clear(BG_COLOR)

        draw_polyline(gui, control_points, POLYGON_COLOR, 1.2)
        draw_control_points(gui)

        if use_bspline:
            curve = build_bspline_polyline()
            draw_polyline(gui, curve, CURVE_COLOR, 2.0)
        else:
            curve = build_bezier_polyline()
            draw_polyline(gui, curve, CURVE_COLOR, 2.0)

        mode_text = "Mode: Cubic B-spline" if use_bspline else "Mode: Bezier"
        gui.text("Left click: add control point", pos=(0.02, 0.96), color=TEXT_COLOR)
        gui.text("C: clear canvas", pos=(0.02, 0.92), color=TEXT_COLOR)
        gui.text("B: switch between Bezier and B-spline", pos=(0.02, 0.88), color=TEXT_COLOR)
        gui.text(mode_text, pos=(0.02, 0.84), color=TEXT_COLOR)
        gui.text(f"Control points: {len(control_points)}", pos=(0.02, 0.80), color=TEXT_COLOR)

        if use_bspline:
            if len(control_points) < 4:
                gui.text("Cubic B-spline needs at least 4 control points", pos=(0.02, 0.76), color=0xAA0000)
            else:
                gui.text("B-spline has local control: moving one point affects only a local part", pos=(0.02, 0.76), color=TEXT_COLOR)
        else:
            gui.text("Bezier has global control: moving one point changes the whole curve", pos=(0.02, 0.76), color=TEXT_COLOR)

        gui.show()


if __name__ == "__main__":
    main()