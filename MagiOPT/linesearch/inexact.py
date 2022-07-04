import torch
from MagiOPT.utils import evaluate

class Armijo:
    def __init__ (self, func, rho=1e-3):
        assert 0 < rho < 1, "rho is not in valid range"        
        self.rho  = rho
        self.func = func
        
    def armijo(self, x, d, alpha):
        grad_x = evaluate(self.func, x)
        right = self.func(x) + self.rho * torch.matmul(grad_x, d) * alpha
        left  = self.func(x + alpha * d)
        return left > right
    
    def backtrack(x, d, self):
        beta = 0.9 # Decay rate
        lam = 1.0
        while self.armijo(x, d, lam):
            lam = beta * lam
        return lam

    def forward(x, d, self):
        Max = 10
        beta = 2
        lam = 0.01
        while self.armijo(x, d, lam) and lam < Max:
            lam = beta * lam
        return lam

            
class Goldstein:
    def __init__ (self, func, rho=1e-3):
        assert 0 < rho < 0.5, "rho is not in valid range"        
        self.rho   = rho
        self.func  = func

    def goldstein(self, alpha):        
        left, right_lower, right_upper = self.conditions(alpha)
        if right_lower <= left <= right_upper:
            return 1
        elif left > right_upper:
            return 2
        else:
            return 3

    def search(self, x, d):
        # Initialization
        self.store(x, d)
        Max = 10
        a = 0
        b = Max
        
        beta = 1.2
        lam = 1
        checker = self.goldstein(lam)
        while checker != 1:
            if checker == 2:
                b = lam
            else:
                a = lam
            if (b - a) < 1e-6:
                break
            lam = (b + a) / 2
            checker = self.goldstein(lam)
        return lam

    def conditions(self, alpha):
        right_lower = self.value + self.rho * self.product * alpha
        right_upper = self.value + (1 - self.rho) * self.product * alpha
        left        = self.func(self.x + alpha * self.d)
        return left, right_lower, right_upper

    def store(self, x, d):
        self.x = x
        self.d = d
        grad_x = evaluate(self.func, x)
        self.product = torch.matmul(grad_x, d)
        self.value   = self.func(x)
 
class Wolfe:
    def __init__ (self, func, rho=1e-3, sigma=1e-2):
        assert 0 < rho < 1, "rho is not in valid range"
        assert rho < sigma < 1, "sigma is not in valid range"        
        self.rho   = rho
        self.sigma = sigma
        self.func  = func
        
    def wolfe(self, alpha, flag):
        
        left, right, temp = self.conditions(alpha)
        x_new = self.x + alpha * self.d
        grad_x_new = evaluate(self.func, x_new)
        
        comp1 = (left <= right)
        # Strong wolfe
        if flag:
            comp2 = abs(torch.matmul(grad_x_new, self.d)) <= -temp
        # Wolfe
        else:
            comp2 = (torch.matmul(grad_x_new, self.d) >= temp)
            
        if comp1 and comp2:
            return 1
        elif not comp1:
            return 2
        else:
            return 3        
        
    def search(self, x, d, flag=False): # flag to be true for "Strong Wolfe" 
        # Initialization
        self.store(x, d)
        Max = 10
        a = 0
        b = Max
        
        beta = 1.2
        lam = 1
        checker = self.wolfe(lam, flag)
        while checker != 1:
            if checker == 2:
                b = lam
            else:
                a = lam
            if (b - a) < 1e-6:
                break
            lam = (b + a) / 2
            checker = self.wolfe(lam, flag)
        return lam        

    def conditions(self, alpha):
        right_lower = self.value + self.rho * self.product * alpha
        right_upper = self.value + (1 - self.rho) * self.product * alpha
        left        = self.func(self.x + alpha * self.d)
        
        left = self.func(self.x + alpha * self.d)

        right = self.value + self.rho * self.product * alpha
        temp = self.sigma * self.product

        return left, right, temp

    def store(self, x, d):
        self.x = x
        self.d = d
        grad_x = evaluate(self.func, x)
        self.product = torch.matmul(grad_x, d)
        self.value   = self.func(x)
