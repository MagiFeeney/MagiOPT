from torch.autograd.functional import jacobian
from MagiOPT.linesearch import exact, Armijo, Goldstein, Wolfe
from MagiOPT.utils import evaluate, criteria, isOutlier
from MagiOPT.visual import plot_surface, plot_contour
import torch
from torch.autograd.functional import jacobian


class LeastSquares:
    def __init__(self, func):
        self.func = func

    def Surrogate(self, x):
        y = 0
        for item in self.func(x):
            y = y + item**2
        return y / 2
        
    def Jacobian(self, x):
        return torch.vstack(jacobian(self.func, inputs=torch.FloatTensor(x)))
    
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
    
class Gaussnewton(LeastSquares):
    def __init__(self, func, ls="Goldstein", sc="gradNorm"):
        super().__init__(func)
        if ls == "exact":
            self.explorer = exact(self.Surrogate)
        else:
            if ls == "Armijo":
                self.explorer = Armijo(self.Surrogate)
            elif ls == "Goldstein":
                self.explorer = Goldstein(self.Surrogate)
            elif ls == "Wolfe":
                self.explorer = Wolfe(self.Surrogate)
            else:
                raise ValueError()
        self.sc = sc
            
    def direction(self, x):
        r = torch.as_tensor(self.func(x))
        J = self.Jacobian(x)
        g = torch.matmul(J.transpose(0, 1), r)
        G = torch.matmul(J.transpose(0, 1), J)
        try:
            G_inv = torch.inverse(G)
        except ValueError:
            print("G is Singular Matrix, Not Invertible")
        return torch.matmul(G_inv, -g)

    def step(self, x0, epsilon=1e-5):
        self.initSeq()
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        if x0.dim() == 0:
            x0 = x0.unsqueeze(0)
        self.sequence.append(x0)
        d = self.direction(x0)
        x1 = x0 + self.explorer.search(x0, d) * d
        isOutlier(x1)
        
        while criteria(self.Surrogate, x0, x1, epsilon, rule=self.sc):
            x0 = x1
            self.sequence.append(x0)
            d = self.direction(x0)
            x1 = x0 + self.explorer.search(x0, d) * d
            isOutlier(x1)
        self.sequence.append(x1)
        return x1

    
class LMF(LeastSquares):
    def __init__(self, func, sc="gradNorm"):
        super().__init__(func)
        self.sc = sc
        
    def direction(self, x, v):
        r = torch.as_tensor(self.func(x))
        J = self.Jacobian(x)
        g = torch.matmul(J.transpose(0, 1), r)
        G = torch.matmul(J.transpose(0, 1), J) + v * torch.eye(J.shape[1])
        try:
            G_inv = torch.inverse(G)
        except ValueError:
            print("G is Singular Matrix, Not Invertible")
        return J, torch.matmul(G_inv, -g)

    def step(self, x0, v=0.1, epsilon=1e-5):
        self.initSeq()
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        if x0.dim() == 0:
            x0 = x0.unsqueeze(0)        
        self.sequence.append(x0)
        J, d = self.direction(x0, v)
        gamma = self.calculateGamma(J, d, x0)
        v = self.adjust(gamma, v)
        if gamma <= 0:
            x1 = x0
        else:
            x1 = x0 + d
        isOutlier(x1)
        
        while criteria(self.Surrogate, x0, x1, epsilon, rule=self.sc):
            x0 = x1
            self.sequence.append(x0)
            J, d = self.direction(x0, v)
            gamma = self.calculateGamma(J, d, x0)
            v = self.adjust(gamma, v)
            if gamma <= 0:
                x1 = x0
            else:
                x1 = x0 + d
            isOutlier(x1)
        self.sequence.append(x1)
        return x1

    def QudraticApprox(self, x, J, d):
        r = torch.as_tensor(self.func(x + d))
        return torch.dot(torch.matmul(d, J.transpose(0, 1)), torch.matmul(J, d)) / 2 + \
            torch.dot(d, torch.matmul(J.transpose(0, 1), r)) + \
            torch.dot(r, r) / 2

    def calculateGamma(self, J, d, x):
        x0 = x
        x1 = x + d
        f1 = self.Surrogate(x0)
        f2 = self.Surrogate(x1)
        q1 = f1
        q2 = self.QudraticApprox(x0, J, d)
        
        delta_f = f2 - f1
        delta_q = q2 - q1

        return delta_f / delta_q
    
    def adjust(self, gamma, v):
        if gamma < 0.25:
            return 4 * v
        elif gamma > 0.75:
            return v / 2
        else:
            return v

        
class Dogleg(LeastSquares):
    def __init__(self, func, sc="gradNorm"):
        super().__init__(func)
        self.sc = sc
        self.Gaussnewton = Gaussnewton(func)
        self.LMF         = LMF(func)

    def step(self, x0, delta=0.1, epsilon=1e-5):
        self.initSeq()
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        if x0.dim() == 0:
            x0 = x0.unsqueeze(0)        
        self.sequence.append(x0)
        J, d = self.direction(x0, delta)
        gamma = self.LMF.calculateGamma(J, d, x0)
        delta = self.adjust(gamma, delta)
        if gamma <= 0:
            x1 = x0
        else:
            x1 = x0 + d
        isOutlier(x1)
        
        while criteria(self.Surrogate, x0, x1, epsilon, rule=self.sc):
            x0 = x1
            self.sequence.append(x0)
            J, d = self.direction(x0, delta)
            gamma = self.LMF.calculateGamma(J, d, x0)
            delta = self.adjust(gamma, delta)
            if gamma <= 0:
                x1 = x0
            else:
                x1 = x0 + d
            isOutlier(x1)
        self.sequence.append(x1)
        return x1      

    def direction(self, x, delta):
        J, Sc = self.Cauchystep(x)
        Sc_norm = torch.norm(Sc)
        if Sc_norm > delta:
            return J, delta * Sc / Sc_norm
        else:
            Sn = self.Newtonstep(x)
            Sn_norm = torch.norm(Sn)
            if Sn_norm <= delta:
                return J, Sc
            else:
                return J, self.linearcomb(Sc, Sn, delta)
        
    def Cauchystep(self, x):
        r = torch.as_tensor(self.func(x))
        J = self.Jacobian(x)
        d = -torch.matmul(J.transpose(0, 1), r)
        Jd = torch.matmul(J, d)
        return J, (torch.norm(d)**2 / torch.norm (Jd)**2) * d

    def Newtonstep(self, x):
        return self.Gaussnewton.direction(x)

    def linearcomb(self, Sc, Sn, delta):
        return torch.dot(Sc, Sc - Sn) / torch.dot(Sn - Sc, Sn - Sc)
    
    def adjust(self, gamma, delta):
        if gamma < 0.25:
            return delta / 4
        elif gamma > 0.75:
            return 2 * delta
        else:
            return delta

## TODO
# QR Decoposition for efficiently solving homogeneous system 


