import numpy as np
import time

class RandomizedGossip:
    """Randomized Gossip Algorithm Class"""
    def __init__(self, adjacency_matrix, initial_values):
        self.n = len(initial_values)
        self.adjacency = adjacency_matrix
        self.values = initial_values.copy()
        self.initial_values = initial_values.copy()
        self.history = [self.values.copy()]
        self.transmissions = 0
        self.transmissions_history = [0]
        self.iterations = 0
        self.true_average = np.mean(initial_values)
        self.neighbors = {}
        for i in range(self.n):
            self.neighbors[i] = np.where(self.adjacency[i])[0]

    def step(self):
        """Execute one Gossip update step"""
        i = np.random.randint(0, self.n)
        if len(self.neighbors[i]) > 0:
            j = np.random.choice(self.neighbors[i])
            avg = (self.values[i] + self.values[j]) / 2
            self.values[i] = avg
            self.values[j] = avg
            self.transmissions += 2
        self.iterations += 1
        self.history.append(self.values.copy())
        self.transmissions_history.append(self.transmissions)

    def run(self, max_iterations=100000, tolerance=1e-8):
        """Run Gossip algorithm until convergence"""
        start_time = time.time()
        for k in range(max_iterations):
            self.step()
            error = np.linalg.norm(self.values - self.true_average) / np.linalg.norm(self.initial_values)
            if error < tolerance:
                elapsed_time = time.time() - start_time
                print(f"Gossip converged at iteration {k+1}")
                print(f"Final error: {error:.2e}")
                print(f"Total transmissions: {self.transmissions}")
                print(f"Runtime: {elapsed_time:.2f} seconds")
                break
            if (k + 1) % 10000 == 0:
                print(f"Gossip iteration {k+1}: error = {error:.2e}")
        return self.history, self.transmissions, self.transmissions_history

    def compute_error_history(self):
        """Compute normalized L2 error history"""
        history_array = np.array(self.history)
        errors = []
        for values in history_array:
            error = np.linalg.norm(values - self.true_average) / np.linalg.norm(self.initial_values)
            errors.append(error)
        return errors