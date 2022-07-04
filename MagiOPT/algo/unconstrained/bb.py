from MagiOPT.linesearch import exact
from MagiOPT.utils import evaluate, criteria, isOutlier
from MagiOPT.visual import plot_surface, plot_contour
import torch


class Barzilai_Borwein:
    def __init__(self, func, sc):
        self.func     = func
        self.sc       = sc
        self.explorer = exact(func)
         
    def step(self, x0, epsilon=1e-5):
        self.initSeq()
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        if x0.dim() == 0:
            x0 = x0.unsqueeze(0)
        self.sequence.append(x0)
        g = evaluate(self.func, x0)
        d = -g
        x1 = x0 + self.magnitude(x0, g) * d
        isOutlier(x1)
        
        while criteria(self.func, x0, x1, epsilon, rule=self.sc):
            cof = self.magnitude(x0, g, x1)
            x0 = x1
            self.sequence.append(x0)
            d  = -evaluate(self.func, x0)
            x1 = x0 + cof * d
            isOutlier(x1)
        self.sequence.append(x1)
        self.updateSeq()
        return x1

    def diff(self, x1, x0, g):
        g1 = evaluate(self.func, x1)
        s = x1 - x0
        y = g1 - g
        return s, y

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

        
class BB1(Barzilai_Borwein):
    def __init__(self, func, sc="gradNorm"):
        super().__init__(func, sc)
        
    def magnitude(self, x0, g, x1=None):
        if x1 != None:
            s, y = self.diff(x1, x0, g)
            sy = torch.dot(s, y)
            yy = torch.dot(y, y)
            if sy < 1e-12 or yy < 1e-12:
                self.updateSeq()
                raise ValueError("Invalid step size, terminate prematurely.")
            return sy / yy
        else:
            return self.explorer.search(x0, -evaluate(self.func, x0))


class BB2(Barzilai_Borwein):
    def __init__(self, func, sc="gradNorm"):
        super().__init__(func, sc)
        
    def magnitude(self, x0, g, x1=None):
        if x1 != None:
            s, y = self.diff(x1, x0, g)
            sy = torch.dot(s, y)
            yy = torch.dot(y, y)
            if sy < 1e-12 or yy < 1e-12:
                self.updateSeq()
                raise ValueError("Invalid step size, terminate prematurely.")
            return sy / yy
        else:
            return self.explorer.search(x0, -evaluate(self.func, x0))    

