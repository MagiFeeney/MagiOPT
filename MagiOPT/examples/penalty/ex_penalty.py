import MagiOPT as optim

def object(x):
  return x[0]**2 + x[1]**2

def constr1(x):
  return x[0] - x[1] + 1

def main():
    sigma = 10
    optimizer = optim.Penalty(object, sigma, (constr1, '<='), plot=True)
    optimizer.BFGS(sc="stepLen") # inner optimizer
    x = optimizer.step([0, 0])
    
if __name__ == "__main__":
    main()
