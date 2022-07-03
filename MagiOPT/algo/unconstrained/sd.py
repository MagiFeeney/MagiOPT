from MagiOPT.linesearch import exact
from MagiOPT.utils import evaluate, criteria, isOutlier
from MagiOPT.visual import plot_surface, plot_contour
import torch


class SD:
    def __init__(self, func, sc="gradNorm"):
        self.func     = func
        self.sc       = sc
        self.explorer = exact(func)
        
    def step(self, x0, epsilon=1e-5):
        self.initSeq()
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        if x0.dim() == 0:
            x0 = x0.unsqueeze(0)
        self.sequence.append(x0)
        d = -evaluate(self.func, x0)
        x1 = x0 + self.explorer.search(x0, d) * d
        isOutlier(x1)
        
        while criteria(self.func, x0, x1, epsilon, rule=self.sc):
            x0 = x1
            self.sequence.append(x0)
            d  = -evaluate(self.func, x0)
            x1 = x0 + self.explorer.search(x0, d) * d
            isOutlier(x1)
        self.sequence.append(x1)
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
