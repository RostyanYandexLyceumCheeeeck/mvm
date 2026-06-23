import math
import numpy as np

from typing import Callable

TF = tuple[float, float]
TC = tuple[complex, complex]


def simpson_method(
        f:        Callable, 
        interval: TF | TC   = (-math.pi/2, math.pi/2), 
        N:        int       = 100, 
        dtype:    type      = np.float64, 
        mach_eps: float     = np.finfo(np.float64).eps,
        ) -> float:

    a = dtype(interval[0] + mach_eps)
    b = dtype(interval[1] - mach_eps)

    k = int(N/2)

    xs, h = np.linspace(a, b, 2*k+1, dtype=dtype, retstep=True)
    ys = np.array([f(x) for x in xs], dtype=dtype)
    
    # [1, 4, 2, 4, 2, ..., 4, 1]
    weights = np.ones(N + 1, dtype=dtype)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2

    return (h / 3) * np.sum(ys * weights)

def gauss_quadrature_method(
        f:        Callable, 
        interval: TF | TC   = (-math.pi/2, math.pi/2), 
        N:        int       = 100, 
        dtype:    type      = np.float64, 
        mach_eps: float     = np.finfo(np.float64).eps,
        ) -> float:
    """
    Классический метод Гаусса. 
    """

    a = dtype(interval[0] + mach_eps)
    b = dtype(interval[1] - mach_eps)

    # узлы t и веса A Лежандра in [-1, 1]
    t, A = np.polynomial.legendre.leggauss(N)
    # print(A)
    
    # scale [-1, 1] to [a, b]
    center = (a + b)/2
    radius = (b - a)/2
    xs = center + t * radius
    ys = np.array([f(x) for x in xs], dtype=dtype)
    
    return radius * np.sum(A * ys)

def composite_gauss_method(
        f:        Callable, 
        interval: TF | TC   = (-math.pi/2, math.pi/2), 
        N:        int       = 100, 
        M:        int       = 0,
        dtype:    type      = np.float64, 
        ) -> float:
        
    """
    Составной метод Гаусса: разбивает интервал на M частей(подинтервалов). 
    На каждом подинтервале применяет классический метод Гаусса, суммирует и возвращает результат. 
    """

    a, b = interval
    sub_edges = np.linspace(a, b, M + 1) # М подинтервалов
    get_sub_interval = lambda i: (sub_edges[i], sub_edges[i+1]) # возвращает i-ый подинтервал

    return sum(gauss_quadrature_method(f, get_sub_interval(i), N=N, mach_eps=0, dtype=dtype) for i in range(M))

def adaptive_integrate(
        method, 
        f, 
        interval, 
        eps=1e-8, 
        p=4, 
        r=2
        ) -> tuple[float, float, int]:
    """
    Адаптивное вычисление интеграла с оценкой погрешности по методу Рунге.
    """

    N = 4
    I_prev = method(f, interval, N, mach_eps=0)
    
    while True:
        N *= r
        I_curr = method(f, interval, N, mach_eps=0)
        
        err = abs(I_curr - I_prev) / ((r ** p) - 1)
        
        if err < eps:
            return I_curr, err, N
        I_prev = I_curr

def func_a(x, eps=1e-10):
    """
        sin(pi * x^5)
       ---------------
           (1-x)x^5

    Особые точки:
        x == 0: предел = pi     через первый замечательный
        x == 1: предел = 5*pi   через Лопиталя
    """
    
    if abs(x) < eps: return math.pi
    if x >= 1 - eps: return 5 * math.pi
    return math.sin(math.pi * x**5) / (x**5 * (1 - x))

def func_b(t, eps=1e-10):
    """
          -√x + sin(x/10)
        e^
    
    Замена x = t/(1-t)
    Особые точки:
        t == 1: предел = 0  
    """
    
    if t >= 1.0 - eps: return 0.0
    
    x = t / (1 - t)
    dx_dt = 1 / (1 - t)**2
    return math.exp(-math.sqrt(x) + math.sin(x/10)) * dx_dt

def test_approximation_order():
    """
    Тестовый пример для метода Симпсона и метода Гаусса: интеграл e^x на отрезке [0, 1].
    Точное значение: e - 1. 
    Порядок аппроксимации составной формулы Симпсона равен 4.
    Порядок аппроксимации формулы Гаусса равен 2N.
    """

    f_test = np.exp
    exact_val = np.exp(1) - 1
    
    k = 2
    interval = (0, 1)

    def test_simpson():
        N1 = 10
        N2 = N1 * k
    
        I1 = simpson_method(f_test, interval, N1, mach_eps=0)
        I2 = simpson_method(f_test, interval, N2, mach_eps=0)
        
        err1 = abs(exact_val - I1)
        err2 = abs(exact_val - I2)
        
        # err1/err2 ~ k^4
        ratio = err1 / err2
        order = math.log(ratio, k)

        print("--- Исследование порядка аппроксимации (Метод Симпсона) ---")
        print(f"Погрешность при N={N1}: {err1:.3e}")
        print(f"Погрешность при N={N2}: {err2:.3e}")
        print(f"Отношение погрешностей: {ratio:.2f} (Ожидается ~ {k**4})")
        print(f"Оцененный порядок точности: {order:.2f}\n")

    def test_gauss():
        N = 2
        M1 = 10
        M2 = M1 * k

        I1 = composite_gauss_method(f_test, interval, N, M1)
        I2 = composite_gauss_method(f_test, interval, N, M2)
        
        err1 = abs(exact_val - I1)
        err2 = abs(exact_val - I2)
        
        # err1/err2 ~ k^(2N)
        ratio = err1 / err2
        order = math.log(ratio, k)
        
        print(f"--- Исследование порядка аппроксимации Метод Гаусса, {N=}) ---")
        print(f"Погрешность при {M1=}: {err1:.8e}")
        print(f"Погрешность при {M2=}: {err2:.8e}")
        print(f"Отношение погрешностей: {ratio:.2f} (Ожидается ~ {k**(2*N)})")
        print(f"Оцененный порядок точности: {order:.2f}\n")

    test_simpson()
    test_gauss()


if __name__ == "__main__":
    test_approximation_order()

    r=2
    eps = 1e-8
    p_gauss = 8
    p_simpson = 4
    interval = (0, 1)

    my_log = lambda b, a: int(math.log(b, a))  # округление вниз от logₐ(b)

    print("--- Вычисление интеграла func_a ---")
    val_a, err_a, n_a = adaptive_integrate(simpson_method, func_a, interval, eps=eps, r=r, p=p_simpson)
    print(f"Метод Симпсона: Ia = {val_a:.9f}, Оценка погрешности: {err_a:.2e}, Узлов: {n_a} = r^{my_log(n_a, r)}")
    
    # Метод Гаусса сходится очень быстро, поэтому порядок в оценке Рунге берем высокий
    val_a_g, err_a_g, n_a_g = adaptive_integrate(gauss_quadrature_method, func_a, interval, eps=eps, r=r, p=p_gauss)
    print(f"Метод Гаусса:   Ia = {val_a_g:.9f}, Оценка погрешности: {err_a_g:.2e}, Узлов: {n_a_g} =  r^{my_log(n_a_g, r)}\n")

    print("--- Вычисление интеграла func_b ---")
    val_b, err_b, n_b = adaptive_integrate(simpson_method, func_b, interval, eps=eps, r=r, p=p_simpson)
    print(f"Метод Симпсона: Ib = {val_b:.9f}, Оценка погрешности: {err_b:.2e}, Узлов: {n_b} = r^{my_log(n_b, r)}")
    
    # Метод Гаусса сходится очень быстро, поэтому порядок в оценке Рунге берем высокий
    val_b_g, err_b_g, n_b_g = adaptive_integrate(gauss_quadrature_method, func_b, interval, eps=eps, r=r, p=p_gauss)
    print(f"Метод Гаусса:   Ib = {val_b_g:.9f}, Оценка погрешности: {err_b_g:.2e}, Узлов: {n_b_g} =  r^{my_log(n_b_g, r)}")



"""
Доделать(сказали на сдаче):
    посчитать порядок аппроксимации на тестовом примере методом гаусса, прислать в тг

Фульман сказал, что порядок гаусса 6    
"""