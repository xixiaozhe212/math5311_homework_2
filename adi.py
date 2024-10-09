import numpy as np
import math
import matplotlib.pyplot as plt

# Constants
PI = math.pi
N_values = [20, 40, 80, 160]  # Different N values
errors = []  # To store maximum errors for each N

def lec5_2d_crank_nicolson_version2(N):
    NX, NY = N, N  # Number of spatial nodes
    L, H = PI, PI      # Spatial range
    TI = 1.0           # Time range
    DX = L / (NX - 1)  # Cell length of X-axis
    DY = H / (NY - 1)  # Cell length of Y-axis
    R = 0.5            # CFL number
    DT1 = R * min(DX, DY)  # Time length
    NT = math.ceil(TI / DT1)  # Time steps
    DT = TI / NT       # Time length
    RX = 0.5 * DT / (DX ** 2)
    RY = 0.5 * DT / (DY ** 2)

    def initial_condition():
        return np.array([[math.sin(i * DX) * math.sin(j * DY) if i < NX - 1 and j < NY - 1 else 0
                          for j in range(NY)] for i in range(NX)])

    def cal_exact():
        return np.array([[math.exp(-2 * TI) * math.sin(i * DX) * math.sin(j * DY) if i < NX - 1 and j < NY - 1 else 0
                          for j in range(NY)] for i in range(NX)])

    def cal_error(u, u_exact):
        return np.max(np.abs(u - u_exact))

    def solve_x_direction(u, u_half):
        A = np.zeros((NX, NX))
        for i in range(1, NX - 1):
            A[i, i - 1] = -RX
            A[i, i] = 1 + 2 * RX
            A[i, i + 1] = -RX

        A[0, 0], A[NX - 1, NX - 1] = 1, 1

        for j in range(1, NY - 1):
            d = u[:, j] + RY * (u[:, j + 1] - 2 * u[:, j] + u[:, j - 1])
            solution = np.linalg.solve(A, d)
            u_half[1:-1, j] = solution[1:-1]

    def solve_y_direction(u_half, u_next):
        A = np.zeros((NY, NY))
        for i in range(1, NY - 1):
            A[i, i - 1] = -RY
            A[i, i] = 1 + 2 * RY
            A[i, i + 1] = -RY

        A[0, 0], A[NY - 1, NY - 1] = 1, 1

        for i in range(1, NX - 1):
            d = np.zeros(NY)  # Ensure d has length NY
            d[1:-1] = u_half[i, 1:-1] + RX * (u_half[i + 1, 1:-1] - 2 * u_half[i, 1:-1] + u_half[i - 1, 1:-1])
            solution = np.linalg.solve(A, d)
            u_next[i, 1:-1] = solution[1:-1]

    def adi_scheme(u):
        u_half = np.zeros_like(u)
        u_next = np.zeros_like(u)

        for _ in range(NT):
            solve_x_direction(u, u_half)
            solve_y_direction(u_half, u_next)
            u[:] = u_next

    # Main logic for the current N
    u = initial_condition()
    u_exact = cal_exact()
    adi_scheme(u)

    return cal_error(u, u_exact)

# Calculate errors for different N values
for N in N_values:
    error = lec5_2d_crank_nicolson_version2(N)
    errors.append(error)

# Plotting
plt.figure()
plt.plot(N_values, errors, marker='o')
plt.xlabel('N (Number of Spatial Nodes)')
plt.ylabel('Maximum Error')
plt.title('Maximum Error vs N')
plt.grid()
plt.xscale('log')  # Optional: logarithmic scale for better visualization
plt.yscale('log')  # Optional: logarithmic scale for better visualization
plt.xticks(N_values)  # Ensure all N values are shown on the x-axis
plt.show()