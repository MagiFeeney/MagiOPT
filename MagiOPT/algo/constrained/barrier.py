import MagiOPT.algo.unconstrained as optim
from MagiOPT.utils import reconstructure
import torch

class Barrier:            
    def step(self, x, eps1=1e-6, epsilon=1e-6):
        assert self.isFeasible(x), "Initial point isn't feasible"
        count = 1
        print(f"round {count}:" )
        x = torch.as_tensor(x, dtype=torch.float32)
        x = self.solver.step(x, eps1)
        if self.plot:
            self.solver.plot({"ba": [self.mu]})        
        while (self.mu * self.criteria(x) > epsilon):
            self.mu /= 10
            count += 1
            print(f"round {count}:" )
            x = self.solver.step(x, eps1)
            if self.plot:
                self.solver.plot({"ba": [self.mu]})            
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
            return 1
        else:
            return 0

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

class inverseBarrier(Barrier):
    def __init__(self, object, mu, *constraints, plot=False):
        super().__init__()
        self.mu = torch.as_tensor(mu, dtype=torch.float32)
        self.object = object
        self.equality, self.inequality = reconstructure(constraints)
        self.plot = plot

    def func(self, x):
        y = 0
        if self.equality != None:
            seq = 0
            for eq in self.equality:
                seq = seq + eq(x)**2
            y = y + seq
                
        if self.inequality != None:
            sneq = 0
            for neq in self.inequality:
                if neq[1]:
                    sneq = sneq + 1 / neq[0](x)
                else:
                    sneq = sneq - 1 / neq[0](x)                    
            y = y + sneq
            
        y = self.mu * y + self.object(x)
        return y

    def criteria(self, x):
        sneq = 0
        if self.inequality != None:
            for neq in self.inequality:
                if neq[1]:
                    sneq = sneq + 1 / neq[0](x)
                else:
                    sneq = sneq - 1 / neq[0](x)

        return sneq 
        
class logBarrier(Barrier):
    def __init__(self, object, mu, *constraints, plot=False):
        super().__init__()
        self.mu = torch.as_tensor(mu, dtype=torch.float32)
        self.object = object
        self.equality, self.inequality = reconstructure(constraints)
        self.plot = plot

    def func(self, x):
        y = 0
        if self.equality != None:
            seq = 0
            for eq in self.equality:
                seq = seq + eq(x)**2
            y = y + seq
                
        if self.inequality != None:
            sneq = 0
            for neq in self.inequality:
                if neq[1]:
                    sneq = sneq + torch.log(neq[0](x))
                else:
                    sneq = sneq + torch.log(-neq[0](x))                    
            y = y + sneq
            
        y = -self.mu * y + self.object(x)
        return y

    def criteria(self, x):
        sneq = 0
        if self.inequality != None:
            for neq in self.inequality:
                if neq[1]:
                    sneq = sneq + torch.log(neq[0](x))
                else:
                    sneq = sneq + torch.log(-neq[0](x))

        return sneq
