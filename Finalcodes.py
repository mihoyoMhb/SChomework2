import numpy as np
import matplotlib.pyplot as plt


def HatFunctions(x, j, xe):
    """Comes from the lab codes, and we do not need
    to modify it"""
    N = np.size(x) - 1
    m = np.size(xe)
    phi = np.zeros(m)
    if 0 < j < N:
        ind1 = np.where((xe >= x[j - 1]) & (xe <= x[j]))[0]
        ind2 = np.where((xe >= x[j]) & (xe <= x[j + 1]))[0]
        phi[ind1] = (xe[ind1] - x[j - 1]) / (x[j] - x[j - 1])
        phi[ind2] = (x[j + 1] - xe[ind2]) / (x[j + 1] - x[j])
    elif j == 0:
        ind = np.where(xe <= x[1])[0]
        phi[ind] = (x[1] - xe[ind]) / (x[1] - x[0])
    elif j == N:
        ind = np.where(xe >= x[N - 1])[0]
        phi[ind] = (xe[ind] - x[N - 1]) / (x[N] - x[N - 1])
    else:
        raise ValueError('Value j must be between 0 and length(x)-1')
    return phi


def MassMatAssembler(x):
    """
    Assembles the mass matrix M based on node coordinates x.
    """
    N = np.size(x) - 1  # number of elements
    n = N - 1  # dimension of the interior nodes
    M = np.zeros((n, n))  # initialize mass matrix to zero

    for i in range(0, n - 1):  # loop over elements
        h = x[i + 1] - x[i]  # element length
        # assemble element mass matrix
        M[i:i + 2, i:i + 2] += (h / 6) * np.array([[2, 1], [1, 2]])

    # Boundary adjustments for mass matrix
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    hn = x[n] - x[n - 1]
    hn1 = x[n + 1] - x[n]
    M[0, 0] = (2 / 3) * h1 + (1 / 3) * h2
    M[n - 1, n - 1] = (1 / 3) * hn + (2 / 3) * hn1

    return M


def StiffMatAssembler(x):
    #
    # Returns the assembled stiffness matrix A
    # Input is a vector x of node coords
    #
    N = np.size(x) - 1  # number of elements
    n = N - 1  # dimension V_h^0 = size(x)-2
    S = np.zeros([n, n])  # initialize stiffnes matrix to zero
    for i in range(0, n - 1):  # loop over elements
        h = x[i + 1] - x[i]  # element length
        # assemble element stiffness
        S[i:i + 2, i:i + 2] = S[i:i + 2, i:i + 2] + np.array([[1, -1], [-1, 1]]) / h

    h1 = x[1] - x[0];
    h2 = x[2] - x[1]
    hn = x[n] - x[n - 1];
    hn1 = x[n + 1] - x[n]
    S[0, 0] = 1 / h1 + 1 / h2  # adjust for left BC
    S[n - 1, n - 1] = 1 / hn + 1 / hn1  # adjust for right BC
    return -S


def LoadVecAssemblerInitial_Analytical(x):
    """Easier to understand how to calculate the results for LoadVec, mathematical analysis is written on my notes"""
    N = np.size(x) - 1  # number of elements
    n = N - 1  # number of unknowns (interior nodes)
    F = np.zeros(n)  # initialize load vector F to zeros
    h = x[1] - x[0]  # assuming uniform mesh

    for i in range(1, N):  # i from 1 to N-1
        xi_minus = x[i - 1]
        xi = x[i]
        xi_plus = x[i + 1]
        F_i = (2 * np.sin(xi) - np.sin(xi_minus) - np.sin(xi_plus)) / h
        F[i - 1] = F_i

    return F


def TimeStepping_explicit(M, S, U0, k, num_steps):
    """Unstable explicit time stepping,
    Large element needed"
    """
    n = len(U0)
    U = U0.copy()
    solutions = {}
    times_for_assignment = [0.0, 0.25, 0.5, 1.0]
    time_points = [k * i for i in range(num_steps + 1)]
    for t in time_points:
        U = U + k * np.linalg.inv(M) @ (S @ U)
        if t in times_for_assignment:
            solutions[t] = U.copy()
    return solutions


def TimeStepping_implicit(M, S, U0, k, num_steps):
    n = len(U0)
    U = U0.copy()
    solutions = {}
    times_for_assignment = [0.0, 0.25, 0.5, 1.0]
    time_points = [k * i for i in range(num_steps + 1)]
    A = M + k * -S
    for t in time_points:
        b = M @ U
        U_new = np.linalg.solve(A, b)
        U = U_new
        if t in times_for_assignment:
            solutions[t] = U.copy()
    return solutions


def TimeStepping_Trapezoidal_method(M, S, U0, k, num_steps):
    """Large element needed"""
    n = len(U0)
    U = U0.copy()
    solutions = {}
    times_for_assignment = [0.0, 0.25, 0.5, 1]
    time_points = [k * i for i in range(num_steps + 1)]
    A = M + k / 2 * -S
    for t in time_points:
        b = (M + k / 2 * S) @ U
        U_new = np.linalg.solve(A, b)
        U = U_new
        if t in times_for_assignment:
            solutions[t] = U.copy()
    return solutions


# Parameters
a, b = 0, np.pi  # interval [a,b] = [0, π]
N = 50  # number of elements
h = (b - a) / N  # mesh size
x = np.linspace(a, b, N + 1)  # node coordinates
n = N - 1  # number of unknowns (interior nodes)
k = 0.0005  # time step size
t_final = 1.0
num_steps = int(np.ceil(t_final / k))
k = t_final / num_steps

# Assemble matrices
M = MassMatAssembler(x)
S = StiffMatAssembler(x)

# Initial condition
F = LoadVecAssemblerInitial_Analytical(x)
U0 = np.linalg.solve(M, F)

# Evaluation points
xe = np.linspace(a, b, 300)
PHI = np.zeros((np.size(xe), n))
for j in range(1, N):
    PHI[:, j - 1] = HatFunctions(x, j, xe)

figure = plt.figure(figsize=(16, 16))
rows, cols = 3, 1
method = {'explicit': TimeStepping_explicit,
          'implicit': TimeStepping_implicit,
          'trapezoid': TimeStepping_Trapezoidal_method, }
for index, (method_name, method) in enumerate(method.items()):
    solutions = {0.0: U0.copy()}
    solutions.update(method(M, S, U0, k, num_steps))

    ax = figure.add_subplot(rows, cols, index + 1)
    for t in sorted(solutions.keys()):
        U = solutions[t]
        uh = PHI @ U
        ax.plot(xe, uh, label=f't = {t:.2f}')
    ax.set_title(f"Numerical Solution ({method_name} Method)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.legend()
    ax.grid()
"""Here show the resulting figure with fixed N = 50 for explicit, implicit, trapezoidal method"""
"""Fixed N V.S. k values for implicit, trapezoidal method"""
# Error vs. Time Step Size (k) with fixed h or N
N = 30  # number of elements
h = (b - a) / N  # mesh size
x = np.linspace(a, b, N + 1)  # node coordinates
n = N - 1  # interior nodes

# Assemble matrices
M = MassMatAssembler(x)
S = StiffMatAssembler(x)

# Initial condition
F = LoadVecAssemblerInitial_Analytical(x)
U0 = np.linalg.solve(M, F)

x_interior = x[1:-1]
t_final = 1.0

# Define decreasing k values
k_values = [_ * 0.01 for _ in range(1, 25)]

# Lists to store errors for each solver
errors_explicit = []
errors_implicit = []
errors_crank_nicolson = []

for k in k_values:
    num_steps = int(np.ceil(t_final / k))
    k = t_final / num_steps
    u_exact = np.exp(-t_final) * np.sin(x_interior)
    # # Explicit Euler method
    # solutions_explicit = TimeStepping_explicit(M, S, U0, k, num_steps)
    # U_explicit = get_solution_at_t(solutions_explicit, t_final)
    #
    # e_explicit = U_explicit - u_exact
    # error_explicit = np.sqrt(e_explicit.T @ M @ e_explicit)
    # errors_explicit.append(error_explicit)

    # Implicit Euler method
    solutions_implicit = TimeStepping_implicit(M, S, U0, k, num_steps)
    # U_implicit = get_solution_at_t(solutions_implicit, t_final)
    U_implicit = solutions_implicit[t_final]
    e_implicit = U_implicit - u_exact
    error_implicit = np.sqrt(e_implicit.T @ M @ e_implicit)
    errors_implicit.append(error_implicit)

    # Crank-Nicolson method
    solutions_cn = TimeStepping_Trapezoidal_method(M, S, U0, k, num_steps)
    #U_cn = get_solution_at_t(solutions_cn, t_final)
    U_cn = solutions_cn[t_final]
    e_cn = U_cn - u_exact
    error_cn = np.sqrt(e_cn.T @ M @ e_cn)
    errors_crank_nicolson.append(error_cn)

# Plotting error vs k
plt.figure()
# plt.loglog(k_values, errors_explicit, '-o', label='Explicit Euler (Ops, very big errors)')
plt.loglog(k_values, errors_implicit, '-o', label='Implicit Euler')
plt.loglog(k_values, errors_crank_nicolson, '-o', label='Trapezoidal method')
plt.xlabel('Time step size k')
plt.ylabel('L2 Error at t = {:.2f}'.format(t_final))
plt.title('Error vs Time Step Size (k)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

"""Fixed N V.S. k values for explicit"""
N = 30
k_values = [_ * 0.00001 for _ in range(1, 25)]
for k in k_values:
    num_steps = int(np.ceil(t_final / k))
    k = t_final / num_steps
    u_exact = np.exp(-t_final) * np.sin(x_interior)
    # Explicit Euler method
    solutions_explicit = TimeStepping_explicit(M, S, U0, k, num_steps)
    #U_explicit = get_solution_at_t(solutions_explicit, t_final)
    U_explicit = solutions_explicit[t_final]
    e_explicit = U_explicit - u_exact
    error_explicit = np.sqrt(e_explicit.T @ M @ e_explicit)
    errors_explicit.append(error_explicit)

# Plotting error vs k
plt.figure()
plt.loglog(k_values, errors_explicit, '-o', label='Explicit Euler (Ops, very big errors)')
# plt.loglog(k_values, errors_implicit, '-o', label='Implicit Euler')
# plt.loglog(k_values, errors_crank_nicolson, '-o', label='Trapezoidal method')
plt.xlabel('Time step size k')
plt.ylabel('L2 Error at t = {:.2f}'.format(t_final))
plt.title('Error vs Time Step Size (k)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

# Error vs Mesh Size (h)

# Fixed time step size
k_fixed = 0.004
t_final = 1.0
num_steps = int(t_final / k_fixed)

# Define increasing N values to get decreasing h
N_values = [20 * k for k in range(1, 10)]
h_values = []
errors_energy1 = []
errors_energy3 = []
for N in N_values:
    h = (b - a) / N
    h_values.append(h)
    x = np.linspace(a, b, N + 1)
    n = N - 1
    M = MassMatAssembler(x)
    S = StiffMatAssembler(x)
    F = LoadVecAssemblerInitial_Analytical(x)
    U0 = np.linalg.solve(M, F)

    # Use the implicit solver
    solutions = TimeStepping_implicit(M, S, U0, k_fixed, num_steps)
    U = solutions[t_final]
    x_interior = x[1:-1]
    u_exact = np.exp(-t_final) * np.sin(x_interior)
    e = U - u_exact
    energy_norm = np.sqrt(e.T @ (-S) @ e)
    errors_energy1.append(energy_norm)

    solutions = TimeStepping_Trapezoidal_method(M, S, U0, k_fixed, num_steps)
    U = solutions[t_final]
    x_interior = x[1:-1]
    u_exact = np.exp(-t_final) * np.sin(x_interior)
    e = U - u_exact
    energy_norm = np.sqrt(e.T @ (-S) @ e)
    errors_energy3.append(energy_norm)

# Plotting error vs h
plt.figure()
plt.loglog(h_values, errors_energy1, '-o', label='implicit Euler')
# plt.loglog(h_values, errors_energy2, '-o', label='Explicit Euler')
plt.loglog(h_values, errors_energy3, '-o', label='trapezoidal method')
plt.xlabel('Mesh size h')
plt.ylabel('Energy Norm of Error at t = {:.2f}'.format(t_final))
plt.title('Error vs Mesh Size (h)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()

# Error vs Mesh Size (h)

# Fixed time step size
k_fixed = 0.0002
t_final = 1.0
num_steps = int(t_final / k_fixed)

# Define increasing N values to get decreasing h
N_values = [k for k in range(10, 50)]
h_values = []
errors_energy1 = []
errors_energy2 = []
errors_energy3 = []
for N in N_values:
    h = (b - a) / N
    h_values.append(h)
    x = np.linspace(a, b, N + 1)
    n = N - 1
    M = MassMatAssembler(x)
    S = StiffMatAssembler(x)
    F = LoadVecAssemblerInitial_Analytical(x)
    U0 = np.linalg.solve(M, F)

    solutions = TimeStepping_explicit(M, S, U0, k_fixed, num_steps)
    U = solutions[t_final]
    x_interior = x[1:-1]
    u_exact = np.exp(-t_final) * np.sin(x_interior)
    e = U - u_exact
    energy_norm = np.sqrt(e.T @ (-S) @ e)
    errors_energy2.append(energy_norm)

# Plotting error vs h
plt.figure()
plt.loglog(h_values, errors_energy2, '-o', label='Explicit Euler')
plt.xlabel('Mesh size h')
plt.ylabel('Energy Norm of Error at t = {:.2f}'.format(t_final))
plt.title('Error vs Mesh Size (h)')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()
