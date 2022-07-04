import MagiOPT as optim

def func(x):
    return (x + 1, 0.5 * x**2 + x - 1)

# or: func = lambda x: (x + 1, 0.5 * x**2 + x - 1)

def main():
    
    optimizer1 = optim.Gaussnewton(func)
    x1 = optimizer1.step(2)
    optimizer1.plot()

    optimizer2 = optim.LMF(func)
    x2 = optimizer2.step(2)
    optimizer2.plot()

    optimizer3 = optim.Dogleg(func)
    x3 = optimizer3.step(2)
    optimizer3.plot()        

if __name__ == "__main__":
    main()
