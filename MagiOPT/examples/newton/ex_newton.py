import MagiOPT as optim

def func(x):
    return 3 * x[0]**2 + 3 * x[1]**2 - x[0]**2 * x[1]

def main():
    
    optimizer = optim.newton(func)
    x = optimizer.step(2)
    optimizer.plot()

if __name__ == "__main__":
    main()
    
