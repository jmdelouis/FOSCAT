
import math

class CircSpline:
    def __init__(self, nodes, degree=3):
        """
        Initialize the circular spline with the given number of nodes and degree.
        """
        self.degree = degree
        self.nodes = nodes
        self.norm = [self._compute_norm(i) for i in range(degree + 1)]

    def _compute_norm(self, i):
        """
        Compute normalization factor for the ith element.
        """
        return pow(-1, i) * (self.degree + 1) / (math.factorial(self.degree + 1 - i) * math.factorial(i))

    def yplus(self, x):
        """
        Compute yplus value for a given x based on the spline's degree.
        """
        if x < 0.0:
            return 0.0
        if self.degree == 0:
            return 0.5 if x == 0.0 else 1.0
        return pow(x, self.degree)

    def calculate(self, x):
        """
        Calculate circular spline values for a given x.
        """
        y = [0] * self.nodes
        for i in range(self.nodes + self.degree // 2 + 1):
            tmp = 0
            tx = self.nodes * math.fmod(x, 2 * math.pi) / (math.pi * 2) - i
            for j in range(self.degree + 1):
                tmp += self.norm[j] * self.yplus(tx - j + (self.degree + 1) // 2)
            if tmp < 0:
                tmp = 0.0
            y[i % self.nodes] += tmp
        for i in range(self.degree // 2):
            tmp = 0
            tx = self.nodes * math.fmod(x, 2 * math.pi) / (math.pi * 2) + 1 + i
            for j in range(self.degree + 1):
                tmp += self.norm[j] * self.yplus(tx - j + (self.degree + 1) // 2)
            if tmp < 0:
                tmp = 0.0
            y[self.nodes - 1 - i] += tmp
        return y

