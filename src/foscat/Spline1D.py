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
        
        
    def cubic_spline_function(self,x):
        """
        Evaluate the cubic spline basis function.
    
        Args:
            x (float or array): Input value(s) to evaluate the spline basis function.
    
        Returns:
            float or array: Result of the cubic spline basis function.
        """
        return -2 * x**3 + 3 * x**2


    def eval(self,x):
        """
        Compute a 3rd-degree cubic spline with 4-point support.
    
        Args:
            x (float or array): Input value(s) to compute the spline.

        Returns:
            indices (array): Indices of the spline support points.
            coefficients (array): Normalized spline coefficients.
        """
        N=self.nodes
        
        if isinstance(x, float):
            # Single scalar input
            base_idx = int(x * (N-1))
            indices = np.zeros([4], dtype="int")
            coefficients = np.zeros([4])
        else:
            # Array input
            base_idx = (x * (N-1)).astype("int")
            indices = np.zeros([4, x.shape[0]], dtype="int")
            coefficients = np.zeros([4, x.shape[0]])

        # Compute the fractional part of the input
        fractional_part = x * (N-1) - base_idx

        # Compute spline coefficients for 4 support points
        coefficients[3] = self.cubic_spline_function(fractional_part / 2) / 2
        coefficients[2] = self.cubic_spline_function(0.5 + fractional_part / 2) / 2
        coefficients[1] = self.cubic_spline_function(1 - fractional_part / 2) / 2
        coefficients[0] = self.cubic_spline_function(0.5 - fractional_part / 2) / 2

        # Assign indices for the support points
        indices[3] = base_idx + 3
        indices[2] = base_idx + 2
        indices[1] = base_idx + 1
        indices[0] = base_idx

        # Handle boundary conditions
        if isinstance(x, float):
            if indices[0] == 0:
                indices[0] = 1
            if indices[1] == 0:
                indices[1] = 1
            if indices[2] == N + 1:
                indices[2] = N
            if indices[3] == N + 1:
                indices[3] = N
            if indices[3] == N + 2:
                indices[3] = N
        else:
            indices[0, indices[0] == 0] = 1
            indices[1, indices[1] == 0] = 1
            indices[2, indices[2] >= N + 1] = N
            indices[3, indices[3] >= N + 1] = N

        # Adjust indices to start from 0
        indices = indices - 1
        # Square coefficients and normalize
        coefficients = coefficients * coefficients
        coefficients /= np.sum(coefficients, axis=0)

        return indices, coefficients
