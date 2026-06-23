import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def euler_explicit(f, y0, t):
    """Явный метод Эйлера 1-го порядка."""

    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        y[i+1] = y[i] + dt * f(t[i], y[i])
    return y

def euler_implicit(f, y0, t):
    """Неявный метод Эйлера 1-го порядка."""
    
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        # нелинейное уравнение: y_{n+1} - y_n - dt * f(t_{n+1}, y_{n+1}) = 0
        func_to_solve = lambda y_next: y_next - y[i] - dt * f(t[i+1], y_next)
        # начальное приближение -- шаг явного метода
        y_guess = y[i] + dt * f(t[i], y[i])
        y[i+1] = fsolve(func_to_solve, y_guess)
    return y

def rk4(f, y0, t):
    """Метод Рунге-Кутты 4-го порядка точности."""

    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        dt = t[i+1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + dt/2, y[i] + dt*k1/2)
        k3 = f(t[i] + dt/2, y[i] + dt*k2/2)
        k4 = f(t[i] + dt, y[i] + dt*k3)
        y[i+1] = y[i] + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return y


def test_approximation_order():
    """
    Исследование порядка аппроксимации. уравнение: y' = -y.
    решение: y(t) = exp(-t).
    """

    def test_f(t, state):
        return np.array([-state[0]])

    def exact_solution(t):
        return np.exp(-t)

    y0_test = np.array([1.0])
    interval = (0, 2)
    N_steps_list = [10, 20, 40, 80, 160] # Разное количество шагов сетки
    steps_h = []
    errors_exp, errors_imp, errors_rk4 = [], [], []

    for N in N_steps_list:
        t_test = np.linspace(*interval, N)
        h = t_test[1] - t_test[0]
        steps_h.append(h)
        
        sol_exp = euler_explicit(test_f, y0_test, t_test)[:, 0]
        sol_imp = euler_implicit(test_f, y0_test, t_test)[:, 0]
        sol_rk4 = rk4(test_f, y0_test, t_test)[:, 0]
        
        # точное решение
        exact = exact_solution(t_test)
        
        # максимальную ошибку по всему отрезку
        errors_exp.append(np.max(np.abs(sol_exp - exact)))
        errors_imp.append(np.max(np.abs(sol_imp - exact)))
        errors_rk4.append(np.max(np.abs(sol_rk4 - exact)))

    # Программно вычислим наклон (порядок p) через линейную регрессию
    p_exp = np.polyfit(np.log(steps_h), np.log(errors_exp), 1)[0]
    p_imp = np.polyfit(np.log(steps_h), np.log(errors_imp), 1)[0]
    p_rk4 = np.polyfit(np.log(steps_h), np.log(errors_rk4), 1)[0]

    print(f"порядок явного Эйлера:   {p_exp:.2f}")
    print(f"порядок неявного Эйлера: {p_exp:.2f}")
    print(f"порядок Рунге-Кутты 4:   {p_rk4:.2f}")

def solution_stiff_system():
    def stiff_system(t, state):
        u, v = state
        du = 998 * u + 1998 * v
        dv = -999 * u - 1999 * v
        return np.array([du, dv])


    t_stiff = np.linspace(0, 0.1, 40) # 40 точек (крупный шаг dt)
    y0_stiff = np.array([1.0, 0.0])   # стартовые значения

    sol_explicit = euler_explicit(stiff_system, y0_stiff, t_stiff)
    sol_implicit = euler_implicit(stiff_system, y0_stiff, t_stiff)

    plt.figure(figsize=(12, 5))

    # График 1: Явный метод (разброс)
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(t_stiff, sol_explicit[:, 0], color='blue', marker='o')
    plt.title('Явный Эйлер')
    plt.ylabel('Значение u')
    plt.grid(True)

    plt.subplot(2, 2, 3, sharex=ax1, sharey=ax1)
    plt.plot(t_stiff, sol_explicit[:, 1], color='orange', marker='o')
    plt.ylabel('Значение v')
    plt.grid(True)


    # График 2: Неявный метод (стабильность)
    ax2 = plt.subplot(2, 2, 2)
    plt.plot(t_stiff, sol_implicit[:, 0], color='blue')
    plt.title('Неявный Эйлер')
    plt.grid(True)

    plt.subplot(2, 2, 4, sharex=ax2, sharey=ax2)
    plt.plot(t_stiff, sol_implicit[:, 1], color='orange')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def solution_predator_prey():
    def predator_prey_system(t, state):
        x, y = state
        a, b = 10, 2
        c, d = 2, 10

        dx = a * x - b * x * y
        dy = c * x * y - d * y
        return np.array([dx, dy])

    t0 = np.linspace(0, 5, 200)

    initial_conditions = [
        [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], 
        [4.0, 4.0], [5.0, 1.0], [6.0, 6.0]
    ]

    plt.figure(figsize=(8, 8))

    for y0 in initial_conditions:
        sol_pp = rk4(predator_prey_system, np.array(y0), t0)
        plt.plot(sol_pp[:, 0], sol_pp[:, 1], label=f'x0={y0[0]}, y0={y0[1]}')

    # особые точки
    plt.scatter([0, 5], [0, 5], color='red', zorder=5, s=50, label='Особые точки')
    plt.annotate('Седло (0,0)', (0.2, 0.2))
    plt.annotate('Центр (5,5)', (5.2, 5.2))

    plt.title('Фазовый портрет модели Хищник-Жертва (Метод РК4)')
    plt.xlabel('Жертвы (x)')
    plt.ylabel('Хищники (y)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_approximation_order()
    solution_stiff_system()
    solution_predator_prey()
