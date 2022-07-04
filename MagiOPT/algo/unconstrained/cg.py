from MagiOPT.linesearch import exact, Armijo, Goldstein, Wolfe
from MagiOPT.utils import evaluate, criteria, isOutlier
from MagiOPT.visual import plot_surface, plot_contour
import torch    


class Conjugate:
    def __init__(self, func, sc):
        self.func     = func
        self.sc       = sc
        
    def step(self, x0, epsilon=1e-5):
        self.initSeq()
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        if x0.dim() == 0:
            x0 = x0.unsqueeze(0)
        self.sequence.append(x0)
        n = x0.shape[0]
        g = evaluate(self.func, x0)
        temp_g = g
        d = -g
        x1 = x0 + self.explorer.search(x0, d) * d
        isOutlier(x1)
        
        counter = 1
        while criteria(self.func, x0, x1, epsilon, rule=self.sc):
            x0 = x1
            self.sequence.append(x0)
            g  = evaluate(self.func, x0)
            if counter % n == 0:
                d = -g
            else:
                d = -g + self.beta(g, temp_g) * d
            temp_g = g
            x1 = x0 + self.explorer.search(x0, d) * d
            isOutlier(x1)
            counter += 1
        self.sequence.append(x1)
        self.updateSeq()
        return x1

    def plot(self, info=None):
        if torch.isnan(self.sequence).any() or torch.isinf(self.sequence).any():
            raise ValueError("Occur \"nan\" or \"inf\" value, cannot proceed.")        
        if self.sequence[0].shape[0] <= 2:
            if info == None:
                plot_surface(self.func, self.sequence)
            else:
                plot_surface(self.func, self.sequence, info)
        if self.sequence[0].shape[0] == 2:
            if info == None:
                plot_contour(self.func, self.sequence)
            else:
                plot_contour(self.func, self.sequence, info)

    def initSeq(self):
        self.sequence = []

    def updateSeq(self):
        self.sequence = torch.vstack(self.sequence)
        

class FR(Conjugate):
    def __init__(self, func, sc="gradNorm"):
        super().__init__(func, sc)
        self.explorer = Wolfe(func)
        
    def beta(self, g, g_prior):
        return torch.dot(g, g) / torch.dot(g_prior, g_prior)

    
class PRP(Conjugate):
    def __init__(self, func, sc="gradNorm"):
        super().__init__(func, sc)
        self.explorer = Wolfe(func)

    def beta(self, g, g_prior):
        return torch.dot(g, g - g_prior) / torch.dot(g_prior, g_prior)

    
class Qudratic:
    def __init__(self, G, b):
        self.G = torch.as_tensor(G, dtype=torch.float32)
        self.b = torch.as_tensor(b, dtype=torch.float32)
        
    def func(self, x):
        return torch.dot(x, torch.matmul(self.G, x)) / 2 + \
            torch.dot(self.b, x)

    def step(self, x0, epsilon=1e-5):
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        if x0.dim() == 0:
            x0 = x0.unsqueeze(0)        
        self.explorer = exact(self.func)
        n = self.G.shape[0] # Termination point
        g = evaluate(self.func, x0)
        temp_g = g
        d = -g
        x1 = x0 + self.explorer.search(x0, d) * d
        for i in range(n - 2):
            x0 = x1
            g  = evaluate(self.func, x0)
            d = -g + self.beta(g, temp_g) * d
            temp_g = g
            x1 = x0 + self.explorer.search(x0, d) * d
        return x1
    
    
class Linearsolver:
    def __init__(self, A, b):
        self.A = torch.as_tensor(A, dtype=torch.float32)
        self.b = torch.as_tensor(b, dtype=torch.float32)

    def step(self, epsilon=1e-5):
        r = self.b
        x = torch.zeros_like(self.b).float()
        p = r
        rdotr = torch.dot(r, r)

        while torch.norm(r) > epsilon:
            Ap = self.Ax(p)
            a = rdotr / torch.dot(p, Ap)
            x += a * p
            r -= a * Ap
            _rdotr = torch.dot(r, r)
            beta = _rdotr / rdotr
            p = r + beta * p
            rdotr = _rdotr
        return x

    def Ax(self, x):
        return torch.matmul(self.A, x)
