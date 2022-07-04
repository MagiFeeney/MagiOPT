import MagiOPT.algo.unconstrained as optim 
from MagiOPT.utils import reconstructure
import torch

class AugLag:
    def __init__(self, object, lam, sigma, *constraints, plot=False):
        self.lam    = torch.as_tensor(lam, dtype=torch.float32)
        self.sigma  = torch.as_tensor(sigma, dtype=torch.float32)
        self.eta    = torch.as_tensor([l / sigma for l in lam], dtype=torch.float32)
        self.object = object
        self.equality, self.inequality = reconstructure(constraints)
        self.plot = plot
        
    def func(self, x):
        index = 0
        seq = 0
        if self.equality != None:
            for eq in self.equality:
                seq = seq - self.lam[index] * eq(x) + self.sigma * eq(x)**2 / 2
                index += 1
                
        sneq = 0
        if self.inequality != None:
            for neq in self.inequality:
                if neq[1]:
                    sneq = sneq + self.sigma * ((torch.min(neq[0](x) - self.eta[index], torch.tensor(0)))**2 - self.eta[index]**2) / 2
                else:
                    sneq = sneq + self.sigma * ((torch.min(-neq[0](x) - self.eta[index], torch.tensor(0)))**2 - self.eta[index]**2) / 2
                index += 1
            
        y = self.object(x) + seq + sneq
        return y
    
    def step(self, x, eps1=1e-6, epsilon=1e-6):
        assert self.isFeasible(x), "Initial point isn't feasible"
        count = 1
        print(f"round {count}:" )        
        x = torch.as_tensor(x, dtype=torch.float32)
        x = self.solver.step(x, eps1)
        if self.plot:
            self.solver.plot({"al": [self.lam, self.sigma]})                
        self.update_lam(x)
        while (self.criteria(x) > epsilon):
            self.sigma *= 10
            self.update_eta()
            count += 1
            print(f"round {count}:" )                             
            x = self.solver.step(x, eps1)
            if self.plot:
                self.solver.plot({"al": [self.lam, self.sigma]})            
            self.update_lam(x)
        return x

    def isFeasible(self, x):
        flag = 1
        if self.equality != None:
            for eq in self.equality:
                if eq(x) != 0:
                    flag = 0

        if self.inequality != None:
            for neq in self.inequality:
                if neq[1]:
                    if neq[0](x) <= 0 or neq[0](x) < 0:
                        flag = 0
                else:
                    if neq[0](x) >= 0 or neq[0](x) > 0:
                        flag = 0

        if flag:
            return 0
        else:
            return 1

    def update_lam(self, x):
        index = 0
        if self.equality != None:
            for eq in self.equality:
                self.lam[index] -= self.sigma * eq(x)
                index += 1

        if self.inequality != None:
            for neq in self.inequality:
                if neq[1]:
                    self.lam[index] = -self.sigma * torch.min(neq[0](x) - self.eta[index], torch.tensor(0))
                else:
                    self.lam[index] = -self.sigma * torch.min(-neq[0](x) - self.eta[index], torch.tensor(0))
                index += 1

    def update_eta(self):
        for i, _ in enumerate(self.eta):
            self.eta[i] = self.lam[i] / self.sigma

    def criteria(self, x):
        index = 0
        seq = 0
        if self.equality != None:
            for eq in self.equality:
                seq = seq + eq(x)**2
                index += 1

        sneq = 0
        if self.inequality != None:
            for neq in self.inequality:
                if neq[1]:
                    sneq = sneq + torch.min(neq[0](x), self.eta[index])**2
                else:
                    sneq = sneq + torch.min(-neq[0](x), self.eta[index])**2
                index += 1
        return torch.sqrt(seq + sneq)

    def SD(self, sc=None):
        if sc == None:
            self.solver = optim.SD(self.func)
        else:
            self.solver = optim.SD(self.func, sc)

    def FR(self, sc=None):
        if sc == None:
            self.solver = optim.FR(self.func)
        else:
            self.solver = optim.FR(self.func, sc)

    def PRP(self, sc=None):
        if sc == None:
            self.solver = optim.PRP(self.func)
        else:
            self.solver = optim.PRP(self.func, sc)

    def newton(self, ls=None, sc=None):
        if ls == None and sc == None:
            self.solver = optim.newton(self.func)
        elif ls != None and sc != None:
            self.solver = optim.newton(self.func, ls, sc)
        elif ls != None and sc == None:
            self.solver = optim.newton(self.func, ls)
        else:
            self.solver = optim.newton(self.func, sc=sc)

    def SR1(self, ls=None, sc=None):
        if ls == None and sc == None:
            self.solver = optim.SR1(self.func)
        elif ls != None and sc != None:
            self.solver = optim.SR1(self.func, ls, sc)
        elif ls != None and sc == None:
            self.solver = optim.SR1(self.func, ls)
        else:
            self.solver = optim.SR1(self.func, sc=sc)

    def DFP(self, ls=None, sc=None):
        if ls == None and sc == None:
            self.solver = optim.DFP(self.func)
        elif ls != None and sc != None:
            self.solver = optim.DFP(self.func, ls, sc)
        elif ls != None and sc == None:
            self.solver = optim.DFP(self.func, ls)
        else:
            self.solver = optim.DFP(self.func, sc=sc)

    def BFGS(self, ls=None, sc=None):
        if ls == None and sc == None:
            self.solver = optim.BFGS(self.func)
        elif ls != None and sc != None:
            self.solver = optim.BFGS(self.func, ls, sc)
        elif ls != None and sc == None:
            self.solver = optim.BFGS(self.func, ls)
        else:
            self.solver = optim.BFGS(self.func, sc=sc)

    def Broyden(self, phi=None, ls=None, sc=None):
        if phi == None and ls == None and sc == None:
            self.solver = optim.Broyden(self.func)
        elif phi == None and ls != None and sc != None:
            self.solver = optim.Broyden(self.func, ls=ls, sc=sc)
        elif phi == None and ls != None and sc == None:
            self.solver = optim.Broyden(self.func, ls=ls)
        elif phi == None and ls == None and sc != None:
            self.solver = optim.Broyden(self.func, sc=sc)
        elif phi != None and ls == None and sc == None:
            self.solver = optim.Broyden(self.func, phi)
        elif phi != None and ls != None and sc != None:
            self.solver = optim.Broyden(self.func, phi, ls, sc)
        elif phi != None and ls != None and sc == None:
            self.solver = optim.Broyden(self.func, phi, ls)
        else:
            self.solver = optim.Broyden(self.func, phi, sc=sc)                        
