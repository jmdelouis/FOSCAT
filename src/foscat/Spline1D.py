
class Spline1D:
    def __init__(self, nodes, degree=3):
        self.degree = degree
        self.nodes = nodes
        self.norm = [0] * (self.degree + 1)
        for i in range(self.degree + 1):
            self.norm[i] = pow(-1, i) * (self.degree + 1) / (self._fact_spline(self.degree + 1 - i) * self._fact_spline(i))

    def _fact_spline(self, x):
        if x <= 1:
            return 1
        return x * self._fact_spline(x - 1)

    def yplus_spline1d(self, x):
        if x < 0.0:
            return 0.0
        if self.degree == 0:
            if x == 0.0:
                return 0.5
            else:
                return 1.0
        return pow(x, self.degree)

    def calculate(self, x):
        y = [0] * self.nodes
        for i in range(self.nodes):
            tmp = 0
            tx = (self.nodes - 1) * x - i
            if x < 0:
                tx = -i
            if x > 1.0:
                tx = (self.nodes - 1) - i
            for j in range(self.degree + 1):
                tmp += self.norm[j] * self.yplus_spline1d(tx - j + (self.degree + 1) / 2)
            if tmp < 0:
                tmp = 0.0
            y[i] += tmp
        total = sum(y)
        y = [yi / total for yi in y]
        return y

