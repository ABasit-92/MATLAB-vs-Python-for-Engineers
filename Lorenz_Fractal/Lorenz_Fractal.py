import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Lorenz system
r = 28
sigma = 10
beta = 8 / 3

# Initial conditions
x1, y1, z1 = 0, 0, 0
x2, y2, z2 = np.sqrt(beta * (r - 1)), np.sqrt(beta * (r - 1)), r - 1
x3, y3, z3 = -np.sqrt(beta * (r - 1)), -np.sqrt(beta * (r - 1)), r - 1

# Grid parameters
nx, nz = 500, 500
xmin, xmax = -40, 40
zmin, zmax = -40, 40

# Generate grid
x_grid = np.linspace(xmin, xmax, nx)
z_grid = np.linspace(zmin, zmax, nz)
X, Z = np.meshgrid(x_grid, z_grid)

# Newton's method parameters
RelTol = 1.e-06
AbsTol = 1.e-09

# Newton's method
for i_x in range(nx):
    
    for j_z in range(nz):
        x = X[j_z, i_x]
        y = 3 * np.sqrt(2)
        z = Z[j_z, i_x]
        error = np.inf

        while error > max(RelTol * max(abs(np.array([x, y, z]))), AbsTol):
            J = np.array([
                [-sigma, sigma, 0],
                [r - z, -1, -x],
                [y, x, -beta]
            ])
            rhs = -np.array([sigma * (y - x), r * x - y - x * z, x * y - beta * z])
            delta_xyz = np.linalg.solve(J, rhs)
            x += delta_xyz[0]
            y += delta_xyz[1]
            z += delta_xyz[2]
            error = max(abs(delta_xyz))

        X[j_z, i_x] = x
        Z[j_z, i_x] = z

# Classify points
eps = 1.e-03
X1 = np.abs(X - x1) < eps
X2 = np.abs(X - x2) < eps
X3 = np.abs(X - x3) < eps
X4 = ~(X1 + X2 + X3)


X = (X1 + 2 * X2 + 3 * X3 + 4 * X4).astype(int)

# Plot the results
plt.figure()
map_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
plt.imshow(X, cmap='tab10', extent=[xmin, xmax, zmin, zmax], origin='lower')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$z$', fontsize=14)
plt.title('Fractal from the Lorenz Equations', fontsize=16)
plt.show()
