import math
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from typing import Callable

TF = tuple[float, float]
TC = tuple[complex, complex]


def mprint(*args, debug: bool, **kwargs):
    if debug: print(*args, **kwargs)

def bisection_solution(
        f:        Callable, 
        interval: TF        = (-math.pi/2, math.pi/2), 
        eps:      float     = 1e-15, 
        dtype:    type      = np.float64,
        mach_eps: float     = np.finfo(np.float64).eps,
        debug:    bool      = False
    ) -> tuple[float, int] | None:
    """
    Поиск корня уравнения "f(t)=0" на заданном интервале методом бисекции(a.k.a. метод деления отрезка пополам).

    **Note**:
        1) Данный метод вернёт *None*, если значения функции около крайних точках заданного интервала одного знака!
        2) Если на заданном интервале несколько корней, то неизвестно, к какому корню сойдётся решение!
        3) Из-за вышеперечисленных пунктов рекомендуем выбирать интервал маленькой длины.

    Args:
        f        (callable):        Заданная функция.
        interval (tuple, optional): Интервал, на котором ищется корень.                Defaults to (-math.pi/2, math.pi/2).
        eps      (float, optional): Требуемая точность.                                Defaults to 1e-15.
        dtype    (type, optional):  Заданный тип, в котором происходят все вычисления. Defaults to np.float64.
        mach_eps (float, optional): Машинный ε заданного типа.                         Defaults to np.finfo(np.float64).eps.
        debug    (bool):            Флаг. если задан, то будут выведены в консоль промежуточные значения, а также сообщения при нестандартной ситуации.
                                                                                       Defaults to False.
    
    :return None:               Если **f(a+mach_eps) * f(b-mach_eps) >= 0** (смотри **Note->п.1**).  
    :return tuple(dtype, int):  Возвращает найденный корень с заданной точностью и количество итераций. значение корня будет в типе *dtype*.
    """

    eps = dtype(eps)
    a, b = dtype(interval[0] + mach_eps), dtype(interval[1] - mach_eps)

    f_temp = f    
    f = lambda t: dtype(f_temp(t))

    if f(a) * f(b) >= 0:
        mprint("Значения функции в крайних точках заданного инервала одного знака!", debug=debug)
        return None
    
    # ----- доп.задание (смотри в конец файла) -----
    
    # print(f"{interval=}, {eps=}")
    # L0 = abs(interval[1] - interval[0])
    # theoretical_n = math.ceil(math.log2(L0 / eps))
    # print(f"Теоретическое количество итераций: {theoretical_n}")
    
    # ----------------------------------------------

    iterations = 0
    c = (a + b) / 2
    while (b - a) / 2 >= eps and f(c) != 0.0:
        iterations += 1
        mprint(c, iterations, debug=debug)
        a, b = (a, c) if f(a) * f(c) < 0 else (c, b) # (a = c) if ... else (b = c)  
        c = (a + b) / 2

    return c, iterations

def simple_iteration_solution(
        f:        Callable, 
        df:       Callable  = None, 
        ddf:      Callable  = None, 
        interval: TF | TC   = (-math.pi/2, math.pi/2), 
        k:        int       = 100, 
        eps:      float     = 1e-15, 
        newton:   bool      = False, 
        dtype:    type      = np.float64, 
        mach_eps: float     = np.finfo(np.float64).eps,
        debug:    bool      = False
    ) -> tuple[float, int] | None:
    """
    Поиск корня уравнения "f(t)=0" на заданном интервале методом простых итераций или методом Ньютона.

    **Note**:
        1) В крайних случаях[когда **df(x) == 0** вблизи **f(x) = 0**] будет использован метод секущих, **только если вычисления вещественные!**
        2) Если интервал задан комплексными числами, то поиск осуществляется в прямоугольнике, где левый нижний угол -- левая граница интервала!
        3) Если длина интервала не превосходит удвоенный машинный ε, то стартовой точкой станет левая граница интервала!
        4) Из предыдущего пункта следует, что для комплекснозначной функции f(z) можно задать стартовую точку 
        5)

    **Caution**: 

    Args:
        f        (callable):           Заданная функция.
        df       (callable, optional): Первая производная f.                                       Defaults to sympy.diff(f).
        ddf      (callable, optional): Вторая производная f.                                       Defaults to sympy.diff(df).
        interval (tuple,    optional): Интервал, на котором ищется корень.                         Defaults to (-math.pi/2, math.pi/2).
        k        (int,      optional): Количество делений интервала.                               Defaults to 100.
        eps      (float,    optional): Требуемая точность.                                         Defaults to 1e-15.
        newton   (bool,     optional): Флаг. если включён, тогда поиск корня идёт методом Ньютона. Defaults to False.
        dtype    (type,     optional): Заданный тип, в котором происходят все вычисления.          Defaults to np.float64.
        mach_eps (float,    optional): Машинный ε заданного типа.                                  Defaults to np.finfo(np.float64).eps.
        debug    (bool,     optional): Флаг. если задан, то будут выведены в консоль промежуточные значения, а также сообщения при нестандартной ситуации.
                                                                                                   Defaults to False.
                                                                                                   
    :return None:               Если корень на промежутке не найден или далее невозможно производить вычисления.
    :return tuple(dtype, int):  Возвращает найденный корень с заданной точностью и количество итераций.   
    """

    # обёртка, чтобы возвращаемое значение было нужного типа
    my_convert = lambda func: (lambda t: dtype(func(t))) 
    is_complex = np.issubdtype(dtype, np.complexfloating)
    
    x_symbol = sp.Symbol('x')
    f_expr = f(x_symbol)
    f = my_convert(f)

    # если крайние точки почти не отличимы друг от друга, то считаем, что интервал задан как (x0, x0)
    if abs(interval[0] - interval[1]) <= 2*mach_eps:
        mach_eps, k = 0, 1

    a = dtype(interval[0] + mach_eps)
    b = dtype(interval[1] - mach_eps)
    
    # создаём массивы узлов сетки и значений функции в этих узлах 
    if is_complex:
        re_vals = np.linspace(a.real, b.real, k)
        im_vals = np.linspace(a.imag, b.imag, k)    
        xs = [dtype(complex(i, j)) for i in re_vals for j in im_vals]
    else:
        xs = np.linspace(a, b, k, dtype=dtype)
    ys = np.array([f(x) for x in xs])

    # находим точку, где функция ближе всего к нулю
    idx = np.argmin(np.abs(ys))
    x0 = xs[idx]

    # вычисляем первую производную, если не задана
    if df is None:
        df_expr = sp.diff(f_expr, x_symbol)
        df = sp.lambdify(x_symbol, df_expr, 'numpy')
    else:
        df_expr = df(x_symbol)
    df = my_convert(df) 

    if df(x0) == 0:
        if not is_complex:
            mprint("Производная вблизи 0 равна df=0. вызывается метод секущих!", debug=debug) 
            return secant_solution(f=f, interval=interval, k=k, eps=eps, dtype=dtype, debug=debug)
        return
     
    if newton:
        # вычисляем вторую производную, если не задана
        if ddf is None:
            ddf = sp.diff(df_expr, x_symbol)
            ddf = sp.lambdify(x_symbol, ddf, 'numpy')
        ddf = my_convert(ddf)

        if not is_complex:
            # чтобы метод сходился, должно выполняться f(x)*f''(x) > 0
            convergence = lambda t: f(t)*ddf(t) > 0 
            if not convergence(x0):
                if len(xs) == 1:
                    mprint("В данной точке нельзя стартовать, т.к. не выполняется \"f(x)*f''(x) > 0\"!", debug=debug) 
                    return
                x0 = xs[idx+1] if (idx+1 < len(xs) and convergence(xs[idx+1])) else xs[idx-1]

    # λ(x0) вычислили заранее, чтобы постоянно не считать 
    lmd = 1/df(x0)
    x_curr, iterations = x0, 0
    while True:
        f_curr = f(x_curr)
        if newton:
            df_curr = df(x_curr)
            if df_curr == 0:
                mprint("Производная ноль, итерация прервана", debug=debug)
                return
            lmd = 1/df_curr

        x_next = x_curr - lmd * f_curr
        
        if not is_complex and k != 1 and (x_next <= a or x_next >= b):
            mprint("Во время поиска корня выскочили за пределы заданного интервала!", debug=debug)
            return

        iterations += 1
        if abs(x_next - x_curr) < eps:
            return x_next, iterations

        mprint(iterations, x_curr, x_next, f_curr, debug=debug)
        x_curr = x_next

def secant_solution(
        f:        Callable, 
        interval: TF        = (-math.pi/2, math.pi/2), 
        k:        int       = 100, 
        eps:      float     = 1e-15, 
        dtype:    type      = np.float64, 
        mach_eps: float     = np.finfo(np.float64).eps,
        debug:    bool      = False
    ) -> tuple[float, int] | None:
    """
    Поиск корня уравнения "f(t)=0" на заданном интервале методом секущих.

    Args:
        f        (callable):        Заданная функция.
        interval (tuple, optional): Интервал, на котором ищется корень.                Defaults to (-math.pi/2, math.pi/2).
        k        (int,   optional): Количество делений интервала.                      Defaults to 100.
        eps      (float, optional): Требуемая точность.                                Defaults to 1e-15.
        dtype    (type,  optional): Заданный тип, в котором происходят все вычисления. Defaults to np.float64.
        mach_eps (float, optional): Машинный ε заданного типа.                         Defaults to np.finfo(np.float64).eps.
        debug    (bool,  optional): Флаг. если задан, то будут выведены в консоль промежуточные значения, а также сообщения при нестандартной ситуации. 
                                                                                       Defaults to False.

    :return None:               Если корень на промежутке не найден. 
    :return tuple(dtype, int):  Возвращает найденный корень с заданной точностью и количество итераций.   
    """

    # обёртка, чтобы возвращаемое значение было нужного типа
    f_temp = f
    f = lambda t: dtype(f_temp(t))

    a = dtype(interval[0] + mach_eps)
    b = dtype(interval[1] - mach_eps)

    # создаём массивы узлов сетки и значений функции в этих узлах 
    xs = np.linspace(a, b, k, dtype=dtype)
    ys = np.array([f(x) for x in xs])
    
    # находим индекс самого близкого к нулю значения
    idx = np.argmin(np.abs(ys))
    x_curr = xs[idx]

    # сосед
    x_prev = xs[idx + 1] if (not idx or ys[idx] < 0) else xs[idx - 1]
    x_next = x_prev
    iterations = 0

    f_prev, f_curr = f(x_prev), f(x_curr)
    while True:
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        mprint(x_prev, x_curr, x_next, iterations, debug=debug)
        if abs(x_next - x_curr) < eps:
            return x_next, iterations

        f_prev, f_curr = f_curr, f(x_next)
        x_prev, x_curr = x_curr, x_next
        iterations += 1

def draw_complex(
        f:          Callable, 
        x_interval: TF        = (-1, 1), 
        y_interval: TF        = (-1, 1),
        eps:        complex   = complex(1e-8, 1e-8)
    ) -> None:
    
    x_range = np.arange(*x_interval, 0.01)
    y_range = np.arange(*y_interval, 0.01)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X + 1j * Y
    
    roots = np.zeros(Z.shape, dtype=np.complex128)
    for (x, y), _ in np.ndenumerate(Z):
        z0 = Z[x, y]
        root, _ = simple_iteration_solution(f=f, interval=(z0, z0), k=1, eps=eps, 
                                            newton=True, dtype=np.complex128, mach_eps=0)
        roots[x, y] = root

    # Identify unique roots and assign a unique color to each
    unique_roots = np.unique(roots.round(decimals=8))
    root_colors = {root: i for i, root in enumerate(unique_roots)}
    colors = np.vectorize(root_colors.get)(roots.round(decimals=8))

    convergent_roots = [i for i in unique_roots]
    print("Roots: ", convergent_roots)

    plt.imshow(colors, extent=(x_interval+y_interval), origin='lower')  # lower чтобы ось Y смотрела вверх
    plt.colorbar()

    # Mark the roots with a red circle
    for root in convergent_roots:
        plt.plot(root.real, root.imag, 'ro') 

    # plt.savefig("lab_02/complex_roots1.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    fun = lambda x: sp.tan(x) - x
    f_poly = lambda z: z**3 - complex(1)
    
    n = 0
    inter = (-math.pi/2 + math.pi * n, math.pi/2 + math.pi * n)

    print(bisection_solution(f=fun, interval=(inter[0] + math.pi/4, inter[1])))
    print(simple_iteration_solution(f=fun, interval=inter, eps=1e-8))
    print(simple_iteration_solution(f=fun, interval=inter, newton=True))
    print(secant_solution(f=fun, interval=inter))
    
    draw_complex(f_poly)

"""
сделать оценку количества итераций для метода бисекций и сравнить с получившимся результатом. отправить в тг !
"""