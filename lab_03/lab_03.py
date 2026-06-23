import math
import numpy as np
import matplotlib.pyplot as plt

from functools import reduce

TI  = tuple[int, int]
TF  = tuple[float, float]
TIF = TI | TF


# n from 3 to 10
# i from 0 to n
f_roots = []


def get_default_Xi(i, n):
    return 2*i/n - 1

def get_chebyshev_Xi(interval: TIF = (-1, 1)):
    
    def chebyshev_Xi(i, n):
        first  = (interval[0] + interval[1]) / 2
        second = (interval[1] - interval[0]) / 2
        third  = math.cos((2*i + 1) / (2*n + 2)* math.pi)
        
        return first + second * third
    
    return chebyshev_Xi

def get_interpolation(f, n, Xi_func):
    global f_roots
    f_roots = []
    for i in range(0, n+1):
        x_i = Xi_func(i, n)
        f_roots.append((f(x_i), x_i))
        
    def lagrange_polinom_support(i, x):
        p = [(x - f_roots[j][1]) / (f_roots[i][1] - f_roots[j][1])  #   П(x - x_j)         /  (x - x_j)  \
                for j in range(0, n+1) if j != i]                   #  ------------  == П |  -----------  |,  j in [0...n]\{i}
        return reduce(lambda x, y: x * y, p)                        #  П(x_i - x_j)        \ (x_i - x_j) /
        
    def lagrange_polinom(x):
        return sum([f_roots[i][0] * lagrange_polinom_support(i, x) for i in range(0, n+1)])
    
    return lagrange_polinom

def draw(f, 
         Xi_func, 
         interval:  TIF  = (-1, 1), 
         losses:    bool = False, 
         title:     str  = "", 
         saved:     bool = False, 
         name_save: str  = "???.png"
         ) -> None:
    
    x = np.linspace(*interval, 10_000)

    for n in range(3, 11):
        P = get_interpolation(f, n, Xi_func)
        Yi_func = P if not losses else lambda i: abs  (P(i) - f(i))

        y = [Yi_func(i) for i in x]

        plt.plot(x, y, label=f"P_{n}(x)")
        # plt.scatter([f_root_list[i][1] for i in range(0, n+1)], [f_root_list[i][0] for i in range(0, n+1)])
    
    # plt.plot(x, [f(i) for i in x], label="f(x)")
    plt.title(title)
    plt.legend()

    if saved:
        plt.savefig(f"lab_03/images/{name_save}", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    flag_save = False
    interval = (-1, 1)
    f = lambda x: 1 / (1 + 25 * x**2)

    # Рисуем обычные полиномы
    draw(f, get_default_Xi, 
         interval=interval, title="Polinoms default", 
         saved=flag_save, name_save="Polinoms_default_draw.png")
    
    # Рисуем полиномы Чебышёва
    draw(f, get_chebyshev_Xi(interval), 
         interval=interval, title="Polinoms chebyshev", 
         saved=flag_save, name_save="Polinoms_chebyshev_draw.png")
    
    # Рисуем функцию ошибки, используя обычные полиномы
    draw(f, get_default_Xi, 
         interval=interval, losses=True, title="Interpolation losses with default nodes", 
         saved=flag_save, name_save="Losses_default.png")
    
    # Рисуем функцию ошибки, используя полиномы Чебышёва
    draw(f, get_chebyshev_Xi(interval), 
         interval=interval, losses=True, title="Interpolation losses with chebyshev nodes", 
         saved=flag_save, name_save="Losses_chebyshev.png")