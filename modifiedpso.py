import numpy as np

# --------------------- Problem Parameters ---------------------
num_tasks = 5  # Number of tasks (n)
num_nodes = 3  # Number of processing nodes (m)
max_iterations = 100  # Number of PSO iterations
population_size = 10  # Number of particles

# Task-specific parameters (Randomly generated for testing)
D_in = np.random.uniform(500, 1000, num_tasks)  # Input data size (bits)
D_out = np.random.uniform(500, 1000, num_tasks)  # Output data size (bits)
TR = np.random.uniform(10, 50, num_tasks)  # Transmission rate (bps)
lambda_ij = np.random.uniform(1, 5, (num_tasks, num_nodes))  # Arrival rate
mu_j = np.random.uniform(6, 10, num_nodes)  # Service rate
T_size = np.random.uniform(1000, 5000, num_tasks)  # Task size (MIPS)
P_cpu = np.random.uniform(500, 2000, num_nodes)  # CPU processing rate

# --------------------- PSO Parameters ---------------------
w_max, w_min = 0.9, 0.4  # Inertia weight limits
c1_max, c1_min = 2.5, 1.5  # Cognitive learning factors
c2_max, c2_min = 2.5, 1.5  # Social learning factors

# --------------------- Initialize Particles ---------------------
X = np.random.randint(0, 2, (population_size, num_tasks, num_nodes))  # Task allocation
V = np.random.uniform(-1, 1, (population_size, num_tasks, num_nodes))  # Velocity
P_best = X.copy()
G_best = X[0].copy()
P_best_fitness = np.full(population_size, np.inf)
G_best_fitness = np.inf

# --------------------- Helper Functions ---------------------
def sigmoid(v):
    """Sigmoid function for discretization."""
    return 1 / (1 + np.exp(-v))

def compute_latency(X):
    """Objective function: Computes total latency (TL) for given task allocation."""
    DT = np.sum(X * ((D_in[:, None] + D_out[:, None]) / TR[:, None]) + (lambda_ij / (2 * mu_j * (mu_j - lambda_ij))))
    Dproc = np.sum(X * (T_size[:, None] / P_cpu[None, :]))
    Dqueue = np.sum(lambda_ij / (2 * mu_j * (mu_j - lambda_ij)))
    return DT + Dproc + Dqueue

# --------------------- PSO Optimization Loop ---------------------
for iteration in range(max_iterations):
    # Compute dynamic inertia weight
    w = w_max - ((w_max - w_min) * iteration / max_iterations) if iteration < 0.7 * max_iterations else \
        w_min + (w_max - w_min) * np.random.rand()

    # Compute adaptive learning factors
    c1 = c1_max + (c1_max - c1_min) * (1 - (np.exp(-w) - 1) ** 2)
    c2 = c2_max + (c2_max - c2_min) * (1 - (np.exp(-w) - 1) ** 2)

    r1, r2 = np.random.rand(), np.random.rand()

    for i in range(population_size):
        # Update velocity
        V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (G_best - X[i])
        V[i] = np.clip(V[i], -1, 1)  # Prevent excessive velocity

        # Update position using sigmoid function
        X[i] = (sigmoid(V[i]) > 0.5).astype(int)

        # Ensure each task is assigned to exactly one node
        for j in range(num_tasks):
            if np.sum(X[i, j]) != 1:  # If task is not assigned correctly
                assigned_idx = np.argmax(V[i, j])  # Assign to highest velocity node
                X[i, j] = 0  # Reset all assignments for this task
                X[i, j, assigned_idx] = 1  # Assign to best node

        # Compute fitness
        fitness = compute_latency(X[i])

        # Update personal best
        if fitness < P_best_fitness[i]:
            P_best[i] = X[i].copy()
            P_best_fitness[i] = fitness

    # Update global best
    best_idx = np.argmin(P_best_fitness)
    if P_best_fitness[best_idx] < G_best_fitness:
        G_best = P_best[best_idx].copy()
        G_best_fitness = P_best_fitness[best_idx]

# --------------------- Output Results ---------------------
print("Best Task Allocation (G_best):")
print(G_best)
print("\nOptimized Total Latency (G_best_fitness):", G_best_fitness)
