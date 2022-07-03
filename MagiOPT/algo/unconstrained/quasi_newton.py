from MagiOPT.linesearch import exact, Armijo, Goldstein, Wolfe
from MagiOPT.utils import evaluate, criteria, isOutlier
from MagiOPT.visual import plot_surface, plot_contour
import torch
from torch.autograd.functional import hessian


class quasinewton:
    def __init__(self, func, ls="exact", sc="gradNorm"):
        if ls == "exact":
            self.explorer = exact(func)
        else:
            if ls == "Armijo":
                self.explorer = Armijo()
            elif ls == "Goldstein":
                self.explorer = Goldstein()
            elif ls == "Wolfe":
                self.explorer = Wolfe()
            else:
                raise ValueError()
        self.func = func
        self.sc   = sc

    def step(self, x0, epsilon=1e-5):
        self.initSeq()
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        if x0.dim() == 0:
            x0 = x0.unsqueeze(0)
        self.sequence.append(x0)
        H = torch.eye(x0.shape[0]) # Initial positive definite matrix 
        g = evaluate(self.func, x0)
        d = -torch.matmul(H, g)
        x1 = x0 + self.explorer.search(x0, d) * d
        isOutlier(x1)
            
        while criteria(self.func, x0, x1, epsilon, rule=self.sc):
            H = self.update(x1, x0, g, H)
            x0 = x1
            self.sequence.append(x0)
            g = evaluate(self.func, x0)
            d = -torch.matmul(H, g)
            x1 = x0 + self.explorer.search(x0, d) * d
            isOutlier(x1)
        self.sequence.append(x1)
        return x1
    
    def diff(self, x1, x0, g):
        g1 = evaluate(self.func, x1)
        s = x1 - x0
        y = g1 - g
        return s.view(-1, 1), y.view(-1, 1)

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


class SR1(quasinewton):
    def __init__(self, func, ls="exact", sc="gradNorm"):
        super().__init__(func, ls, sc)
        
    def update(self, x1, x0, g, H):
        s, y = self.diff(x1, x0, g)
        numerator   = torch.mm(s - torch.mm(H, y), (s - torch.mm(H, y)).transpose(0, 1))
        denominator = torch.mm((s - torch.mm(H, y)).transpose(0, 1), y)
        H = H + numerator / denominator
        return H


class DFP(quasinewton):
    def __init__(self, func, ls="exact", sc="gradNorm"):
        super().__init__(func, ls, sc)
           
    def update(self, x1, x0, g, H):
        s, y = self.diff(x1, x0, g)
        Hy = torch.mm(H, y)
        yH = torch.mm(y.transpose(0, 1), H)
        term1 = torch.mm(s, s.transpose(0, 1)) / torch.mm(s.transpose(0, 1), y)
        term2 = torch.mm(Hy, yH) / torch.mm(y.transpose(0, 1), Hy)
        H = H + term1 - term2
        return H


class BFGS(quasinewton):
    def __init__(self, func, ls="exact", sc="gradNorm"):
        super().__init__(func, ls, sc)
            
    def update(self, x1, x0, g, H):
        s, y = self.diff(x1, x0, g)
        Hy = torch.mm(H, y)
        yH = torch.mm(y.transpose(0, 1), H)             
        term1 = torch.mm(y.transpose(0, 1), Hy) / torch.mm(y.transpose(0, 1), s)
        term2 = torch.mm(s, s.transpose(0, 1)) / torch.mm(y.transpose(0, 1), s)
        term3 = (torch.mm(s, yH) + torch.mm(Hy, s.transpose(0, 1))) / torch.mm(y.transpose(0, 1), s)
        H = H + (1 + term1) * term2 - term3
        return H

    
class Broyden(quasinewton):
    def __init__(self, func, phi=0.5, ls="exact", sc="gradNorm"):
        super().__init__(func, ls, sc)
        self.phi = phi
        self.DFP = DFP(func)
        self.BFGS = BFGS(func)
            
    def update(self, x1, x0, g, H):
        H_DFP  = self.DFP.update(x1, x0, g, H)
        H_BFGS = self.BFGS.update(x1, x0, g, H)
        H = (1 - self.phi) * H_DFP + self.phi * H_BFGS
        return H
                
