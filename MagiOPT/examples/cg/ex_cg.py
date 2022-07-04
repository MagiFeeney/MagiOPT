import MagiOPT as optim

def func1(x):
    return 3 * x[0]**2 + 3 * x[1]**2 - x[0]**2 * x[1]

'''
def func2(x):
   return 3 * x[0]**2 + 3 * x[1]**2 - x[0] * x[1]
We can write func2 as a qudratic form such that with
G = [[6, -1],
     [-1, 6]]
therefore have f = 1/2 x^{T} G x
'''

G = [[6, -1],
     [-1, 6]]

b1 = [0, 0]


# Linear equation, which has a unique solution x^{*} = [11/14, 1/7]
A = [[2, 0],
     [0, 0]]

b2 = [2, 1]

def main():
    
    optimizer1 = optim.FR(func1)
    x1 = optimizer1.step(2)
    optimizer1.plot()

    optimizer2 = optim.PRP(func1)
    x2 = optimizer2.step(2)
    optimizer2.plot()

    optimizer3 = optim.Qudratic(G, b1)
    x3 = optimizer3.step(2)
    
    optimizer4 = optim.Linearsolver(A, b2)
    x4 = optimizer4.step() # Don't need initial point    
            
if __name__ == "__main__":
    main()
