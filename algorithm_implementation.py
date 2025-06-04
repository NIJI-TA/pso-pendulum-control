import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Параметры маятника
J, m, g, h = 1.0, 1.0, 9.81, 0.5
# Параметры для другого маятника J, m, g, h = 0.05, 0.5, 9.81, 0.25

# Максимальное управление
M = 4.0  
# M = 2.0

# Число интервалов управления
N = 20

# Параметры PSO
w, c1, c2 = 0.7, 1.4, 1.4
# Количество частиц
S = 100

# Число итераций
max_iter = 100
# Интеравал вычислений
tau = 1.4
t_max = 10.0

# Создаем массивы частиц (использую матрицу S на N) и их скоростей - случайных чисел с равномерным распределением
particles = np.random.uniform(0, M, (S, N))
velocities = np.random.uniform(-1, 1, (S, N))
# Локальное лучшее для частицы — это она сама на текущий момент
pbest = particles.copy()
value_at_pbest = np.full(S, np.inf)
# Глобальное лучшее пока просто нули, на первой итерации сразу перезапишется
gbest = np.zeros(N)
# Значение целевой функции в gbest
value_at_gbest = np.inf
# Массив сходимисти (для графика сходимости)
convergence = []


# Функция моделирования маятника
def simulate_pendulum(U, t_max, dt=0.02, method="RK45"):
    # Промежутки вычислений
    times = np.arange(0, t_max + dt, dt)
    # Интервал действия компоненты управления
    control_interval = t_max / N

    # Вычисление правых частей системы дифференциальных уравнений для solve_ivp
    def dynamics(t, state):
        # Угол phi и скорость omega подаются как начальное состояние state
        phi, omega = state
        # Вычислим какая компонента управления должна действовать в данных момент времени t
        index = int(t // control_interval)
        index = min(index, N - 1)
        # Берем действующую в момент t компоненту управления
        u = U[index]
        # Первое уравнение системы (уравнение на производную phi)
        dphi = omega
        # Второе уравнение системы (уравнение на производную omega)
        domega = (-m * g * h * np.sin(phi) + u) / J
        # Возвращаем полученные правые части системы
        return [dphi, domega]

    # Решаем систему используя функцию solve_ivp
    result = solve_ivp(
        dynamics, [0, t_max], [np.pi / 2, 0], t_eval=times, method=method
    )
    # Возвращаем массив решений result.y и массив времени result.t
    return result.y, result.t


# Считаем целевую функцию L
def function(U, tau):
    # y - массив решений, t - массив времени
    y, t = simulate_pendulum(U, t_max)
    # phi - угол, omega - угловая скорость
    phi, omega = y
    # шаг по времени
    dt = t[1] - t[0]

    # Распределенное запаздывание (чтобы не хранить ошибку при начальных t)
    start_index = int(tau / dt)

    L = np.sum(phi[start_index:] ** 2) * dt 

    return L


# PSO
# Запускаем главный цикл, используем tqdm для вывода шкалы процесса
for i in tqdm(range(max_iter), desc="Итерации", unit="итерация"):
    # Идем по каждой частице
    for j in range(S):
        # Моделируем маятник для управления particles[j] с N компонентами и запаздыванием tau
        current_function = function(particles[j], tau)
        # Ищем индивидуально лучшую минимизацию L
        if current_function < value_at_pbest[j]:
            # Eсли текущее значение меньше индивидуально лучшего, то запоминаем значение и управление
            value_at_pbest[j] = current_function
            pbest[j] = particles[j].copy()
        # Ищем глобально лучшую минимизацию L
        if current_function < value_at_gbest:
            # Eсли текущее значение меньше глобально лучшего, то запоминаем значение и управление
            value_at_gbest = current_function
            gbest = particles[j].copy()
    # Добавляем в массив для графика сходимости лучшее глобальное значение L на итерации i
    convergence.append(value_at_gbest)
    # Обновление скоростей и позиций согласно методу
    r1 = np.random.rand(S, N)
    r2 = np.random.rand(S, N)
    velocities = (
        w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
    )
    particles = particles + velocities
    # Ограничение управления
    particles = np.clip(particles, 0, M)


print("Лучшее управление:", gbest)
print("Лучшая минимизация:", value_at_gbest)