import numpy as np
import time

class PDMM:
    """Primal-Dual Method of Multipliers (PDMM) Algorithm Class"""
    def __init__(self, adjacency_matrix, initial_values, c=0.1, mode='average',
                 broadcast=False, transmission_loss=0, active_nodes=1, theta_mode='1/2-avg'):
        self.n = len(initial_values)
        self.initial_values = initial_values.copy()
        self.c = c
        self.mode = mode
        self.broadcast = broadcast
        self.transmission_loss = transmission_loss
        self.active_nodes = active_nodes

        # Set theta based on theta_mode
        if theta_mode == '1/2-avg':
            self.theta = 0.5  # ADMM equivalent
        elif theta_mode == 'std':
            self.theta = 1.0  # Standard PDMM
        else:
            raise ValueError("theta_mode must be '1/2-avg' or 'std'")

        self.theta_mode = theta_mode

        if self.mode == 'average':
            self.true_target = np.mean(initial_values)
        elif self.mode == 'median':
            self.true_target = np.median(initial_values)
        else:
            raise ValueError("Mode must be 'average' or 'median'")

        self.iterations = 0
        self.history = []
        self.transmissions = 0
        self.transmissions_history = []

        self.adj_matrix = adjacency_matrix
        self.degrees = np.sum(self.adj_matrix, axis=1)

        self.values = initial_values.copy()

        if self.mode == 'average':
            self.z = {(i, j): np.zeros(1) for i in range(self.n) for j in range(self.n) if self.adj_matrix[i, j] == 1}
            self.y = {(i, j): np.zeros(1) for i in range(self.n) for j in range(self.n) if self.adj_matrix[i, j] == 1}
        elif self.mode == 'median':
            self.z = {(i, j): 0 for i in range(self.n) for j in range(self.n) if self.adj_matrix[i, j] == 1}
            self.y = {(i, j): 0 for i in range(self.n) for j in range(self.n) if self.adj_matrix[i, j] == 1}

        self.neighbors = []
        for i in range(self.n):
            neighbors_idx = []
            for j in range(self.n):
                if self.adj_matrix[i, j] == 1:
                    neighbors_idx.append(j)
            self.neighbors.append(neighbors_idx)

    def get_active_nodes(self):
        """Get list of active nodes for current iteration"""
        if self.active_nodes < 1:
            num_active = int(self.active_nodes * self.n)
            return np.random.choice(range(self.n), num_active, replace=False)
        else:
            return range(self.n)

    def calculate_error(self, x_est, x_true):
        """Calculate normalized L2 error"""
        norm_x_true = np.linalg.norm(x_true)
        if norm_x_true == 0:
            return np.linalg.norm(x_est - x_true)
        return np.linalg.norm(x_est - x_true) / norm_x_true

    def step_average(self):
        """Execute one PDMM iteration for average computation"""
        active_nodes = self.get_active_nodes()
        transmission_count = 0
        active_nodes_set = set(active_nodes)
        new_values = self.values.copy()
        for i in active_nodes:
            sum_Az = np.zeros(1)
            for j in self.neighbors[i]:
                A_ij = 1 if i < j else -1
                sum_Az += A_ij * self.z[(i, j)]
            new_values[i] = (self.initial_values[i] - sum_Az[0]) / (1 + self.c * len(self.neighbors[i]))
        self.values = new_values
        y_prev = self.y.copy()
        for i in active_nodes:
            for j in self.neighbors[i]:
                A_ij = 1 if i < j else -1
                self.y[(i, j)] = self.z[(i, j)] + 2 * self.c * A_ij * self.values[i]
        new_z = self.z.copy()
        for i in active_nodes:
            if self.broadcast:
                if self.transmission_loss > 0 and np.random.rand() < self.transmission_loss:
                    transmission_count += 1
                    continue
                active_neighbors = [j for j in self.neighbors[i] if j in active_nodes_set]
                if active_neighbors:
                    for j in active_neighbors:
                        A_ij = 1 if i < j else -1
                        new_z[(j, i)] = y_prev[(j, i)] + 2 * self.c * A_ij * self.values[i]
                    transmission_count += 1
            else:
                for j in self.neighbors[i]:
                    if j not in active_nodes_set:
                        continue
                    if self.transmission_loss > 0 and np.random.rand() < self.transmission_loss:
                        transmission_count += 1
                        continue
                    new_z[(j, i)] = self.y[(i, j)]
                    transmission_count += 1
        self.z = new_z
        self.transmissions += transmission_count
        self.iterations += 1
        self.history.append(self.values.copy())
        self.transmissions_history.append(self.transmissions)

    def step_median(self):
        """Execute one PDMM iteration for median computation - UNICAST ONLY"""
        active_nodes = self.get_active_nodes()
        transmission_count = 0
        for i in active_nodes:
            neighbors_idx = self.neighbors[i]
            if len(neighbors_idx) == 0:
                continue
            sum_val = 0
            for j in neighbors_idx:
                A = 1 if i < j else -1
                sum_val += A * self.z.get((i, j), 0)
            val = -sum_val / (self.c * len(neighbors_idx))
            bound = 1 / (self.c * len(neighbors_idx))
            if np.abs(self.initial_values[i] - val) <= bound:
                self.values[i] = self.initial_values[i]
            else:
                self.values[i] = val + np.sign(self.initial_values[i] - val) * bound
            for j in neighbors_idx:
                A = 1 if i < j else -1
                self.y[(i, j)] = self.z.get((i, j), 0) + 2 * self.c * A * self.values[i]
        z_prev = self.z.copy()
        for i in active_nodes:
            neighbors_idx = self.neighbors[i]
            for j in neighbors_idx:
                A = 1 if i < j else -1
                self.z[(j, i)] = (1 - self.theta) * z_prev.get((j, i), 0) + self.theta * self.y.get((i, j), 0)
                transmission_count += 1
        self.transmissions += transmission_count
        self.iterations += 1
        self.history.append(self.values.copy())
        self.transmissions_history.append(self.transmissions)

    def step(self):
        """Execute one PDMM iteration based on the selected mode"""
        if self.mode == 'average':
            self.step_average()
        elif self.mode == 'median':
            self.step_median()

    def run(self, max_iterations=1000, tolerance=1e-8):
        """Run PDMM algorithm until convergence"""
        start_time = time.time()
        x_true = self.true_target * np.ones(self.n)
        for k in range(max_iterations):
            self.step()
            error = self.calculate.error(self.values, x_true)
            if error < tolerance:
                elapsed_time = time.time() - start_time
                print(f"PDMM ({self.mode}, θ={self.theta}, broadcast={self.broadcast}) converged at iteration {k+1}")
                print(f"Final error: {error:.2e}")
                print(f"Total transmissions: {self.transmissions}")
                print(f"Runtime: {elapsed_time:.2f} seconds")
                break
            if (k + 1) % 100 == 0:
                print(f"PDMM ({self.mode}, θ={self.theta}, broadcast={self.broadcast}) iteration {k+1}: error = {error:.2e}")
        return self.history, self.transmissions, self.transmissions_history

    def compute_error_history(self):
        """Compute normalized L2 error history"""
        history_array = np.array(self.history)
        errors = []
        x_true = self.true_target * np.ones(self.n)
        for values in history_array:
            error = self.calculate_error(values, x_true)
            errors.append(error)
        return errors