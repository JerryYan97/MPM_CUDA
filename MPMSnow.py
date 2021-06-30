import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 90000 * quality ** 2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-10
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
process = False

material = ti.field(dtype=int, shape=n_particles)  # material id
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F_elasticity = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
F_plasticity = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())


@ti.func
def FixedCorotatedStress(F, mu, lam):
    J = F.determinant()
    U, sig, V = ti.svd(F)
    R = U @ V.transpose()
    return 2.0 * mu * (F - R) @ F.transpose() + lam * (J - 1.0) * J * ti.Matrix([[1, 0], [0, 1]])


@ti.func
def BSpline1D(x):
    abs_x = abs(x)
    res = 0.0
    if 0 <= abs_x < 0.5:
        res = 0.75 - abs_x ** 2
    elif 0.5 <= abs_x < 1.5:
        res = 0.5 * (1.5 - abs_x) ** 2
    else:
        res = 0.0
    return res


@ti.func
def BSplineInterpolation(xp, xg, dx):
    tmp = (xp - xg) / dx
    return BSpline1D(tmp[0]) * BSpline1D(tmp[1])


@ti.func
def BSplineDerivative(x):
    res = 0.0
    if -0.5 < x < 0.5:
        res = -2.0 * x
    elif 0.5 <= x < 1.5:
        res = x - 1.5
    elif -1.5 < x <= -0.5:
        res = 1.5 + x
    else:
        res = 0.0
    return res


@ti.func
def GradWip(xp, xg, dx):
    i = (xp - xg) / dx
    return ti.Vector([
        BSplineDerivative(i[0]) * BSpline1D(i[1]) / dx,
        BSpline1D(i[0]) * BSplineDerivative(i[1]) / dx
    ])


@ti.kernel
def substep():
    # 1. Clean grid data
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0.0
    # 2. P2G
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        Jp = F_plasticity[p].determinant()
        h = ti.exp(10.0 * (1.0 - Jp))
        # h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp))))
        tau = FixedCorotatedStress(F_elasticity[p], h * mu_0, h * lambda_0)
        t_w = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            grid_idx = base + offset
            grid_pos = grid_idx.cast(float) * dx
            w = BSplineInterpolation(x[p], grid_pos, dx)
            dpos = grid_pos - x[p]
            grid_v[base + offset] += w * p_mass * (v[p] + C[p] @ dpos)
            grid_m[base + offset] += w * p_mass
            grid_v[base + offset] += -p_vol * dt * tau @ GradWip(x[p], grid_pos, dx)

            t_w += w
        if abs(t_w - 1.0) > 0.0001:
            print("Wrong with weight:", t_w)
    # 3. Velocity update
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] = grid_v[i, j] / grid_m[i, j]
            grid_v[i, j] += dt * gravity[None] * 9.8  # gravity
            if i <= 5 or i >= n_grid - 5 or j <= 5 or j >= n_grid - 5:
                grid_v[i, j][0] = 0
                grid_v[i, j][1] = 0
        else:
            grid_v[i, j] = [0, 0]
            grid_m[i, j] = 0.0
    # 4. G2P
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        grad_v = ti.Matrix.zero(float, 2, 2)
        t_w = 0.0
        for i, j in ti.static(ti.ndrange(3, 3)):  # loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            grid_idx = base + offset
            grid_pos = grid_idx.cast(float) * dx
            dpos = grid_pos - x[p]
            g_v = grid_v[base + offset]
            w = BSplineInterpolation(x[p], grid_pos, dx)
            new_v += g_v * w
            new_C += 4.0 * w * g_v.outer_product(dpos) / (dx * dx)
            grad_v += g_v.outer_product(GradWip(x[p], grid_pos, dx))
            t_w += w
        if abs(t_w - 1.0) > 0.0001:
            print("Wrong with weight:", t_w)
        Fe_n = F_elasticity[p]
        Fp_n = F_plasticity[p]
        Fe_np1_trial = (ti.Matrix([[1, 0], [0, 1]]) + dt * grad_v) @ Fe_n
        U, sig, V = ti.svd(Fe_np1_trial)
        # sig_star1 = max(0.99, min(1.01, sig[0, 0]))
        # sig_star2 = max(0.99, min(1.01, sig[1, 1]))
        sig_star1 = min(max(sig[0, 0], 1 - 8e-2), 1 + 1.5e-2)
        sig_star2 = min(max(sig[1, 1], 1 - 8e-2), 1 + 1.5e-2)
        sig_star = ti.Matrix([[sig_star1, 0], [0, sig_star2]])
        Fe_np1 = U @ sig_star @ V.transpose()
        Fp_np1 = Fe_np1.inverse() @ Fe_np1_trial @ Fp_n
        v[p] = new_v
        C[p] = new_C
        x[p] += dt * v[p]  # advection
        F_elasticity[p] = Fe_np1
        F_plasticity[p] = Fp_np1


@ti.kernel
def reset():
    group_size = n_particles // 1
    for i in range(n_particles):
        x[i] = [ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size) + 0.2, ti.random() * 0.2 + 0.05 + 0.2]
        v[i] = [0, -6.4]
        material[i] = 2  # 0: fluid 1: jelly 2: snow
        F_elasticity[i] = ti.Matrix([[1, 0], [0, 1]])
        F_plasticity[i] = ti.Matrix([[1, 0], [0, 1]])
        C[i] = ti.Matrix.zero(float, 2, 2)


print("[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse bottons to attract/repel. Press R to reset.")
gui = ti.GUI("Taichi MLS-MPM-128", res=512, background_color=0x112F41)
reset()
gravity[None] = [0, -1]

for frame in range(20000):
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'r':
            reset()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    if gui.event is not None: gravity[None] = [0, 0]  # if had any event
    if gui.is_pressed(ti.GUI.LEFT, 'a'): gravity[None][0] = -1
    if gui.is_pressed(ti.GUI.RIGHT, 'd'): gravity[None][0] = 1
    if gui.is_pressed(ti.GUI.UP, 'w'): gravity[None][1] = 1
    if gui.is_pressed(ti.GUI.DOWN, 's'): gravity[None][1] = -1
    mouse = gui.get_cursor_pos()
    gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
    for s in range(int(2e-3 // dt)):
        substep()
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
    color = colors[material.to_numpy()]
    gui.circles(x.to_numpy(), radius=0.5, color=color)
    gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk