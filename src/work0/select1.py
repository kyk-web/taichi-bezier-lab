import math
import taichi as ti

ti.init(arch=ti.gpu)

WIDTH = 800
HEIGHT = 600
POINT_COLOR = 0xFF3030
POLYGON_COLOR = 0x808080
TEXT_COLOR = 0x111111

MAX_CTRL_POINTS = 256
MAX_SAMPLES = 8192

control_points = []

curve_img = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
sample_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_SAMPLES)
sample_count = ti.field(dtype=ti.i32, shape=())


@ti.kernel
def clear_canvas():
    for i, j in curve_img:
        curve_img[i, j] = ti.Vector([1.0, 1.0, 1.0])


@ti.kernel
def draw_curve_antialias():
    green = ti.Vector([0.0, 0.78, 0.18])

    for k in range(sample_count[None]):
        px = sample_points[k][0]
        py = sample_points[k][1]

        cx = ti.cast(px, ti.i32)
        cy = ti.cast(py, ti.i32)

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                ix = cx + dx
                iy = cy + dy

                if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
                    pixel_center_x = ti.cast(ix, ti.f32) + 0.5
                    pixel_center_y = ti.cast(iy, ti.f32) + 0.5

                    dist = ti.sqrt(
                        (pixel_center_x - px) * (pixel_center_x - px)
                        + (pixel_center_y - py) * (pixel_center_y - py)
                    )

                    w = max(0.0, 1.0 - dist / 1.5)
                    w = w * w

                    old = curve_img[ix, iy]
                    curve_img[ix, iy] = old * (1.0 - w) + green * w


def bezier_point(points, t):
    work = [[p[0], p[1]] for p in points]
    n = len(work)
    for r in range(1, n):
        for i in range(n - r):
            work[i][0] = (1.0 - t) * work[i][0] + t * work[i + 1][0]
            work[i][1] = (1.0 - t) * work[i][1] + t * work[i + 1][1]
    return work[0][0], work[0][1]


def rebuild_curve_samples():
    n = len(control_points)
    if n < 2:
        sample_count[None] = 0
        return

    approx_len = 0.0
    for i in range(n - 1):
        x1, y1 = control_points[i]
        x2, y2 = control_points[i + 1]
        approx_len += math.hypot(x2 - x1, y2 - y1)

    samples = int(max(200, min(MAX_SAMPLES, approx_len * 1.2 + 200)))

    for i in range(samples):
        t = i / (samples - 1) if samples > 1 else 0.0
        x, y = bezier_point(control_points, t)
        sample_points[i] = ti.Vector([x * WIDTH, y * HEIGHT])

    sample_count[None] = samples


def add_control_point(x, y):
    if len(control_points) >= MAX_CTRL_POINTS:
        return
    control_points.append((x, y))
    rebuild_curve_samples()


def clear_all():
    control_points.clear()
    sample_count[None] = 0


def draw_control_polygon(gui):
    if len(control_points) < 2:
        return
    for i in range(len(control_points) - 1):
        gui.line(
            begin=control_points[i],
            end=control_points[i + 1],
            color=POLYGON_COLOR,
            radius=1.2,
        )


def draw_control_points(gui):
    for p in control_points:
        gui.circle(pos=p, radius=5.0, color=POINT_COLOR)


def main():
    gui = ti.GUI("Select 1 - Anti-Aliased Bezier", res=(WIDTH, HEIGHT))
    prev_left = False

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in ["c", "C"]:
                clear_all()

        left_now = gui.is_pressed(ti.GUI.LMB)
        if left_now and not prev_left:
            mx, my = gui.get_cursor_pos()
            add_control_point(mx, my)
        prev_left = left_now

        clear_canvas()
        if sample_count[None] > 0:
            draw_curve_antialias()

        gui.set_image(curve_img)

        draw_control_polygon(gui)
        draw_control_points(gui)

        gui.text("Left click: add control point", pos=(0.02, 0.96), color=TEXT_COLOR)
        gui.text("C: clear canvas", pos=(0.02, 0.92), color=TEXT_COLOR)
        gui.text(
            "Anti-aliasing: distance-weighted 3x3 neighborhood blending",
            pos=(0.02, 0.88),
            color=TEXT_COLOR,
        )
        gui.text(f"Control points: {len(control_points)}", pos=(0.02, 0.84), color=TEXT_COLOR)

        gui.show()


if __name__ == "__main__":
    main()