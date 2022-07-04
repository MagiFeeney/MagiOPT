import MagiOPT as optim

def func(x):
    return 3 * x[0]**2 + 3 * x[1]**2 - x[0] * x[1]

def main():
    # SR1
    optimizer1 = optim.SR1(func)
    x1 = optimizer1.step([0.2, 2])
    optimizer1.plot()

    # DFP
    optimizer2 = optim.DFP(func)
    x2 = optimizer2.step([0.2, 2])
    optimizer2.plot()

    # BFGS
    optimizer3 = optim.BFGS(func)
    x3 = optimizer3.step([0.2, 2])
    optimizer3.plot()

    # Broyden
    optimizer4 = optim.Broyden(func, phi=0.5)
    x4 = optimizer4.step([0.2, 2])
    optimizer4.plot()
    

if __name__ == "__main__":
    main()
