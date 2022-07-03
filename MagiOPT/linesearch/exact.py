from MagiOPT.utils.backforth import backforth
from MagiOPT.utils.goldensection import gss

class exact:
    def __init__(self, func):
        self.func    = func
        self.a       = 0
        self.b       = 10

    def search(self, x, d, step=0.1, epsilon=1e-6):
        # Initial guess
        l, r = backforth(self.func, x, d, self.a, self.b, step)
        astar = gss(self.func, x, d, l, r, epsilon)

        return astar
