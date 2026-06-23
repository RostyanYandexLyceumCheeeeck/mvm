import numpy as np
import matplotlib.pyplot as plt

DEBUG = False


def algorithm_TDMA(a, b, c, d):
    """Метод прогонки для системы: a_i*y_{i-1} + b_i*y_i + c_i*y_{i+1} = d_i"""
    n = len(d)
    y = np.zeros(n)
    
    b_mod = np.copy(b)
    d_mod = np.copy(d)

    if DEBUG:
        print(a)
        print(b_mod)
        print(c)
        print(d_mod)

    # Прямой ход
    for i in range(1, n):
        xi = a[i] / b_mod[i-1]
        b_mod[i] = b_mod[i] - xi * c[i-1]
        d_mod[i] = d_mod[i] - xi * d_mod[i-1]
    
    if DEBUG:
        print(a)
        print(b_mod)
        print(c)
        print(d_mod)

    # Обратный ход
    y[-1] = d_mod[-1] / b_mod[-1]
    for i in range(n-2, -1, -1):
        y[i] = (d_mod[i] - c[i] * y[i+1]) / b_mod[i]
        
    return y


def solve_universal(N, left_type, left_val, right_type, right_val):
    """
    Решает y'' = sin(x) на (0, pi) с произвольными ГУ.
    left_type, right_type: 'dirichlet' или 'neumann'
    left_val, right_val: числовые значения граничных условий
    """
    tau = np.pi / N
    x = np.linspace(0, np.pi, N+1)
    
    # y_{i-1} - 2y_i + y_{i+1} = tau^2 * sin(x_i)
    a = np.ones(N+1)
    b = -2.0 * np.ones(N+1)
    c = np.ones(N+1)
    d = (tau**2) * np.sin(x)
    
    # левое ГУ
    if left_type == 'dirichlet':
        b[0] = 1.0
        c[0] = 0.0
        d[0] = left_val
    elif left_type == 'neumann':
        b[0] = -1.0
        c[0] = 1.0
        d[0] = tau * left_val + (tau**2 / 2.0) * np.sin(x[0])
    else:
        raise ValueError("Тип ГУ должен быть 'dirichlet' или 'neumann'")

    # правое ГУ
    if right_type == 'dirichlet':
        a[-1] = 0.0
        b[-1] = 1.0
        d[-1] = right_val
    elif right_type == 'neumann':
        a[-1] = -1.0
        b[-1] = 1.0
        d[-1] = tau * right_val - (tau**2 / 2.0) * np.sin(x[-1])
    else:
        raise ValueError("Тип ГУ должен быть 'dirichlet' или 'neumann'")

    y = algorithm_TDMA(a, b, c, d)
    return x, y


def solve_fft(N):
    """
    Решает y'' = sin(x) на [0, 2pi) с периодическими ГУ
    используя Быстрое преобразование Фурье (FFT).
    """
    x = np.linspace(0, 2*np.pi, N, endpoint=False)
    f = np.sin(x)
    
    f_hat = np.fft.fft(f)
    
    # Так как длина интервала L = 2pi, то частоты: 0, 1, 2, ..., -2, -1
    k = np.fft.fftfreq(N) * N
    
    # -k^2 * y_hat = f_hat
    y_hat = np.zeros_like(f_hat)
    
    non_zero = (k != 0)
    y_hat[non_zero] = -f_hat[non_zero] / (k[non_zero]**2)
    y_hat[0] = 0.0 
    
    y = np.fft.ifft(y_hat).real
    
    return x, y


def exact_dirichlet(x):
    """Аналитическое решение для y'' = sin(x), y(0)=0, y(pi)=0"""
    return -np.sin(x)

def exact_neumann_mixed(x):
    """Аналитическое решение для y'' = sin(x), y'(0)=2, y(pi)=0"""
    return -np.sin(x) + 3*(x - np.pi)

def exact_fft(x):
    """Аналитическое решение для периодической задачи"""
    return -np.sin(x)


def test_approximation_order():
    """Анализ порядка сходимости (ошибки)"""
    N_sizes = [16, 32, 64, 128, 256, 512]
    
    err_dirichlet = []
    err_neumann = []
    err_fft = []
    
    print("Исследование порядка аппроксимации:")
    print(f"{'N':<8}{'Err Дирихле':<15}{'Err Нейман':<15}{'Err FFT':<15}")
    print("-" * 55)
    
    for N in N_sizes:
        # Error прогонки (Дирихле-Дирихле: 0, 0)
        xd, ud = solve_universal(N, 'dirichlet', 0.0, 'dirichlet', 0.0)
        ed = np.max(np.abs(ud - exact_dirichlet(xd)))
        err_dirichlet.append(ed)
        
        # Error прогонки (Нейман-Дирихле: 2, 0)
        xn, un = solve_universal(N, 'neumann', 2.0, 'dirichlet', 0.0)
        en = np.max(np.abs(un - exact_neumann_mixed(xn)))
        err_neumann.append(en)
        
        # Error фурье (FFT)
        xf, uf = solve_fft(N)
        ef = np.max(np.abs(uf - exact_fft(xf)))
        err_fft.append(ef)
        
        print(f"{N:<8}{ed:<15.2e}{en:<15.2e}{ef:<15.2e}")
        
    # порядок сходимости
    for i in range(len(N_sizes) - 1):
        p_dir = np.log(err_dirichlet[i]/err_dirichlet[i+1]) / np.log(2)
        p_neu = np.log(err_neumann[i]/err_neumann[i+1]) / np.log(2)
        print(f"Переход N={N_sizes[i]}->{N_sizes[i+1]}: Порядок Дирихле = {p_dir:.2f}, Порядок Неймана = {p_neu:.2f}")

    plt.figure(figsize=(8, 6))
    plt.loglog(N_sizes, err_dirichlet, 'o-', label='М. прогонки (Дирихле) ~ $\mathcal{O}(h^2)$')
    plt.loglog(N_sizes, err_neumann, 's-', label='М. прогонки (Нейман-Дирихле) ~ $\mathcal{O}(h^2)$')
    plt.loglog(N_sizes, err_fft, '^-', label='Метод FFT (спектральная точность)')
    
    ref_line = [err_dirichlet[0] * (N_sizes[0]/n)**2 for n in N_sizes]
    plt.loglog(N_sizes, ref_line, 'k--', alpha=0.5, label='Эталонный наклон $\mathcal{O}(h^2)$')
    
    plt.title("Зависимость максимальной ошибки от размера сетки $N$")
    plt.xlabel("Количество узлов сетки $N$")
    plt.ylabel("Максимальная абсолютная ошибка $L_\infty$")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def many_tests():
    global DEBUG
    N = 64

    # Вариант 1: Дирихле-Дирихле: y(0)=0, y(pi)=0 -> Решение: y = -sin(x)
    x1, y1 = solve_universal(N, 'dirichlet', 0.0, 'dirichlet', 0.0)

    # Вариант 2: Дирихле-Нейман: y(0)=0, y'(pi)=1 -> Решение: y = -sin(x)
    x2, y2 = solve_universal(N, 'dirichlet', 0.0, 'neumann', 1.0)

    # Вариант 3: Нейман-Дирихле: y'(0)=2, y(pi)=0 -> Решение: y = -sin(x) + 3*(x - pi)
    x3, y3 = solve_universal(N, 'neumann', 2.0, 'dirichlet', 0.0)


    plt.figure(figsize=(10, 6))
    plt.plot(x1, y1, 'r-', lw=2, label="y(0)=0, y(pi)=0")
    plt.plot(x2, y2, 'b--', lw=2, label="y(0)=0, y'(pi)=1")
    plt.plot(x3, y3, 'g-.', lw=2, label="y'(0)=2, y(pi)=0")


    # ----------------------------------------------------------------------------
    # Вариант 4: Нейман-Нейман: y'(0)=0, y'(pi)=2 -> Решение: y = x - sin(x) + C
    # ответ: не сможет решить, т.к. либо не будет решений, 
    # либо бесконечно много(т.к. С может быть любой). 
    # т.е. последний элемент на главной диагонали будет = 0, и метод сломается.

    # DEBUG = True
    # x4, y4 = solve_universal(3, 'neumann', 0.0, 'neumann', 2.0)
    # plt.plot(x4, y4, 'y-x', lw=2, label="y'(0)=0, y'(pi)=2")
    # ----------------------------------------------------------------------------

    plt.title("Уравнение y'' = sin(x) с различными граничными условиями")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    many_tests()
    test_approximation_order()

