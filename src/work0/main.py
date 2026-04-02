import numpy as np
import taichi as ti

# 基本配置
WINDOW_SIZE = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100

# 颜色定义
BACKGROUND_COLOR = ti.Vector([1.0, 1.0, 1.0])
CURVE_COLOR = ti.Vector([0.0, 1.0, 0.0])
POLYGON_COLOR = ti.Vector([0.6, 0.6, 0.6])

# UI 圆点颜色
POINT_COLOR = (1.0, 0.0, 0.0)

# 按实验要求优先使用 GPU
ti.init(arch=ti.gpu)

# GPU 字段
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WINDOW_SIZE, WINDOW_SIZE))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)


# 清屏
@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = BACKGROUND_COLOR

# 画贝塞尔曲线
@ti.kernel
def draw_curve_kernel(n):
    for i in range(n):
        p = curve_points_field[i]
        x = int(p[0] * (WINDOW_SIZE - 1))
        y = int(p[1] * (WINDOW_SIZE - 1))

        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                px = x + dx
                py = y + dy
                if 0 <= px < WINDOW_SIZE and 0 <= py < WINDOW_SIZE:
                    pixels[px, py] = CURVE_COLOR

# 画控制多边形
@ti.kernel
def draw_polygon_kernel(n):
    for i in range(n - 1):
        p0 = gui_points[i]
        p1 = gui_points[i + 1]

        x0 = p0[0] * (WINDOW_SIZE - 1)
        y0 = p0[1] * (WINDOW_SIZE - 1)
        x1 = p1[0] * (WINDOW_SIZE - 1)
        y1 = p1[1] * (WINDOW_SIZE - 1)

        dx = x1 - x0
        dy = y1 - y0
        steps = int(max(abs(dx), abs(dy))) + 1

        for s in range(2000):
            if s < steps:
                t = 0.0
                if steps > 1:
                    t = s / (steps - 1)

                x = int(x0 + t * dx)
                y = int(y0 + t * dy)

                for ox in ti.static(range(-1, 2)):
                    for oy in ti.static(range(-1, 2)):
                        px = x + ox
                        py = y + oy
                        if 0 <= px < WINDOW_SIZE and 0 <= py < WINDOW_SIZE:
                            pixels[px, py] = POLYGON_COLOR


# =========================
# De Casteljau 算法（CPU）
# =========================
def de_casteljau(points, t):
    temp = np.array(points, dtype=np.float32)
    n = len(temp)

    while n > 1:
        temp[: n - 1] = (1.0 - t) * temp[: n - 1] + t * temp[1:n]
        n -= 1

    return temp[0]


def build_curve_points(control_points):
    curve_np = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    for i in range(NUM_SEGMENTS + 1):
        t = i / NUM_SEGMENTS
        curve_np[i] = de_casteljau(control_points, t)
    return curve_np


def build_gui_points(control_points):
    gui_np = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    for i, p in enumerate(control_points):
        gui_np[i] = p
    return gui_np

def main():
    window = ti.ui.Window("Bezier Curve - De Casteljau", (WINDOW_SIZE, WINDOW_SIZE))
    canvas = window.get_canvas()

    control_points = []

    clear_pixels()
    gui_points.from_numpy(build_gui_points(control_points))

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.LMB:
                if len(control_points) < MAX_CONTROL_POINTS:
                    x, y = window.get_cursor_pos()
                    control_points.append([x, y])
            elif e.key == "c" or e.key == "C":
                control_points.clear()

        clear_pixels()

        gui_np = build_gui_points(control_points)
        gui_points.from_numpy(gui_np)

        if len(control_points) >= 2:
            draw_polygon_kernel(len(control_points))

        if len(control_points) >= 2:
            control_np = np.array(control_points, dtype=np.float32)
            curve_np = build_curve_points(control_np)
            curve_points_field.from_numpy(curve_np)
            draw_curve_kernel(NUM_SEGMENTS + 1)

        canvas.set_image(pixels)
        canvas.circles(gui_points, radius=0.008, color=POINT_COLOR)

        window.show()


if __name__ == "__main__":
    main()