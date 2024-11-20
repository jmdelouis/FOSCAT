import numpy as np

class Spline1D:
    def __init__(self, nodes, degree=3):
        """
        Initializes the Spline1D instance.

        Parameters:
        - nodes (int): The number of nodes in the spline.
        - degree (int): The degree of the spline. Default is 3.
        """
        self.degree = degree
        self.nodes = nodes
        self.norm = np.array([
            pow(-1, i) * (self.degree + 1) /
            (self._fact_spline(self.degree + 1 - i) * self._fact_spline(i))
            for i in range(self.degree + 1)
        ])

    def _fact_spline(self, x):
        """
        Computes the factorial of x.

        Parameters:
        - x (int): Input value.

        Returns:
        - int: The factorial of x.
        """
        if x <= 1:
            return 1
        return x * self._fact_spline(x - 1)

    def yplus_spline1d(self, x):
        """
        Computes the spline function for positive x values.

        Parameters:
        - x (np.ndarray or float): Input value(s).

        Returns:
        - np.ndarray or float: Spline function value(s).
        """
        if self.degree == 0:
            return np.where(x == 0.0, 0.5, 1.0)
        return np.maximum(0, x) ** self.degree

    def calculate(self, x):
        """
        Computes the spline weights for a single x value.

        Parameters:
        - x (float): Input x value in the range [0, 1].

        Returns:
        - np.ndarray: Array of spline weights for each node.
        """
        y = [0] * self.nodes
        for i in range(self.nodes):
            tmp = 0
            tx = (self.nodes - 1) * x - i
            if x < 0:
                tx = -i
            if x > 1.0:
                tx = (self.nodes - 1) - i
            for j in range(self.degree + 1):
                tmp += self.norm[j] * self.yplus_spline1d(
                    tx - j + (self.degree + 1) / 2
                )
            if tmp < 0:
                tmp = 0.0
            y[i] += tmp
        total = sum(y)
        y = [yi / total for yi in y]
        return y

    def eval(self, x_array):
        """
        Evaluates the spline weights for an array of x values in a vectorized manner.

        Parameters:
        - x_array (np.ndarray): Array of x values in the range [0, 1].

        Returns:
        - tuple: (indices, weights)
            - indices (np.ndarray): 2D array where each row contains indices of non-zero weights for each x in x_array.
            - weights (np.ndarray): 2D array where each row contains the non-zero weights for each x in x_array.
        """
        y = np.zeros([self.nodes]+list(x_array.shape))
        
        for i in range(self.nodes):
            tmp = 0
            tx = (self.nodes - 1) * x_array - i
            tx[x_array<0]=-i
            tx[x_array>1.0]=(self.nodes - 1) - i
            
            for j in range(self.degree + 1):
                tmp += self.norm[j] * self.yplus_spline1d(
                    tx - j + (self.degree + 1) / 2
                )
            tmp[tmp<0.0] = 0.0
            y[i] += tmp
        nshape=x_array.shape[0]
        for k in range(1,len(x_array.shape)):
            nshape*=x.shape[k]
        total = sum(y.reshape(self.nodes,nshape),1)
        return y/total
