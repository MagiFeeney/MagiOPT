import MagiOPT as optim

def object(x):
    return x[0]**2 + x[1]**2

def constr1(x):
    return x[0] - x[1] + 1

def main():
    mu = 0.1
    optimizer1 = optim.logBarrier(object, mu, (constr1, '<='), plot=True)
    optimizer1.BFGS(sc="stepLen") # inner optimizer
    x1 = optimizer1.step([-5, 5])

    mu = 0.1
    optimizer2 = optim.inverseBarrier(object, mu, (constr1, '<='), plot=True)
    optimizer2.BFGS(sc="stepLen") # inner optimizer
    x2 = optimizer2.step([-5, 5])        
    
if __name__ == "__main__":
    main()
