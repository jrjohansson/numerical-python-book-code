
def fib(n): 
    """ 
    Return a list of the first n Fibonacci numbers.
    """ 
    f0, f1 = 0, 1
    f = [1] * n 
    for m in range(1, n):
        f[m] = f0 + f1
        f0, f1 = f1, f[m]

    return f

print(fib(10))
