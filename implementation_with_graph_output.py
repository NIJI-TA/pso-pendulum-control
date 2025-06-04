import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
import time

start_time = time.time()

# Параметры маятника
J, m, g, h = 1.0, 1.0, 9.81, 0.5  # J, m, g, h = 0.05, 0.5, 9.81, 0.25
# Максимальное управление
M = 1.5  # M = 2.0
# Число интервалов управления
N = 20
# Параметры PSO
w, c1, c2 = 0.7, 1.4, 1.4
# Количество частиц
S = 200
# Число итераций
max_iter = 150
# Интеравал вычислений
tau = 4.5
t_max = 15.0

# создаем массивы частиц (использую матрицу S на N) и их скоростей - случайных чисел с равномерным распределением
particles = np.random.uniform(0, M, (S, N))
velocities = np.random.uniform(-1, 1, (S, N))
# локальное лучшее для частицы — это она сама на текущий момент
pbest = particles.copy()
value_at_pbest = np.full(S, np.inf)
# глобальное лучшее пока просто нули, на первой итерации сразу перезапишется
gbest = np.zeros(N)
# значение целевой функции в gbest
value_at_gbest = np.inf
# массив сходимисти (для графика сходимости)
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


end_time = time.time()
time_spent = end_time - start_time


print("Лучшее управление:", gbest)
print("Лучшая минимизация:", value_at_gbest)
print("Время выполнения:", time_spent)

# Получение динамики для gbest
y, t = simulate_pendulum(gbest, t_max)
phi, omega = y

# Динамика для нулевого управления
zero = np.zeros(N)
zero_y, zero_t = simulate_pendulum(zero, t_max)
zero_phi, zero_omega = zero_y

# Графики
plt.style.use("default")
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 4)

# ======== Данные маятника, управления, интервала и pso
info = (
    f"Параметры маятника: "
    f"J = {J:.3f}, m = {m:.3f}, h = {h:.3f};\n\n"
    f"Число интервалов управления: N = {N};\n"
    f"Ограничение управления: M = {M:.3f};\n\n"
    f"Интервал от tau = {tau} до T_max = {t_max};\n\n\n"
    f"Параметры PSO:\n"
    f"  Число частиц: {S}, итераций: {max_iter};\n\n\n"
    f"Оптимальное управление (u):\n\n{np.round(gbest, 6)};\n\n\n"
    f"Лучшая минимизация: {value_at_gbest}."
)
# Добавляем в верхний левый угол
ax0 = fig.add_subplot(gs[0, 0:2])
ax0.axis("off")
ax0.text(
    0,
    1,
    info,
    va="top",
    ha="left",
    family="monospace",
    fontsize=8,
    bbox=dict(boxstyle="square", facecolor="white", edgecolor="black", linewidth=1),
)


# ======== Рисуем график динамики маятника без управления
ax1 = fig.add_subplot(gs[0, 2:4])
ax1.plot(
    t, zero_phi, color="slategray", linestyle="--", linewidth=0.7, label="φ (угол)"
)
ax1.set_title("Динамика маятника без управления", fontsize=9)
ax1.set_xlabel("Время, с", fontsize=8)
ax1.set_ylabel("Значения φ", fontsize=8)
# Уменьшаем дефолтный шаг с 0.5 до 0.2 для основных делений по оси y
ax1.yaxis.set_major_locator(MultipleLocator(0.2))
# с 1 до 0.5 для основных делений по оси x
ax1.xaxis.set_major_locator(MultipleLocator(0.5))
ax1.grid(True, linestyle=":", linewidth=0.5)
ax1.legend(fontsize=9)
ax1.set_xlim(left=0)
ax1.set_ylim(np.min(zero_phi) * 1.1, np.max(zero_phi) * 1.1)
ax1.tick_params(labelsize=6)


# ======== Рисуем график динамики маятника с управлением
ax2 = fig.add_subplot(gs[1:2, 0:2])
ax2.plot(t, phi, color="black", linewidth=1, label="φ (угол)")
ax2.set_title("Динамика маятника c управлением", fontsize=9)
ax2.set_xlabel("Время, с", fontsize=8)
ax2.set_ylabel("Значения φ", fontsize=8)
# Уменьшаем дефолтный шаг с 0.5 до 0.2 для основных делений по оси y
ax2.yaxis.set_major_locator(MultipleLocator(0.2))
# с 1 до 0.5 для основных делений по оси x
ax2.xaxis.set_major_locator(MultipleLocator(0.5))
ax2.grid(True, linestyle=":", linewidth=0.5)
ax2.legend(fontsize=9)
ax2.set_xlim(left=0)
ax2.set_ylim(
    min(np.min(phi), np.min(omega)) * 1.1, max(np.max(phi), np.max(omega)) * 1.1
)
ax2.tick_params(labelsize=6)


# ======== Рисуем график угловой скорости
ax3 = fig.add_subplot(gs[1:2, 2:4])
ax3.plot(t, omega, color="gray", linewidth=1, label="ω (угловая скорость)")
ax3.plot(
    t,
    zero_omega,
    color="slategray",
    linestyle=":",
    linewidth=1,
    label="ω при нулевом управлении",
)
ax3.set_xlabel("Время, с", fontsize=9)
ax3.set_ylabel("Значения ω", fontsize=9)
# Уменьшаем дефолтный шаг с 0.5 до 0.2 для основных делений по оси y
ax3.yaxis.set_major_locator(MultipleLocator(0.5))
# с 1 до 0.5 для основных делений по оси x
ax3.xaxis.set_major_locator(MultipleLocator(0.5))
ax3.grid(True, linestyle=":", linewidth=0.5)
ax3.legend(fontsize=9)
ax3.set_xlim(left=0)
ax3.set_ylim(
    min(np.min(phi), np.min(omega)) * 1.1, max(np.max(phi), np.max(omega)) * 1.1
)
ax3.tick_params(labelsize=3)


# ======== Рисуем график кусочно-постоянного управления
ax4 = fig.add_subplot(gs[2:3, 0:2])
u_time = np.linspace(0, t_max, N + 1)
gbest_step = np.append(gbest, gbest[-1])
ax4.step(
    u_time, gbest_step, where="post", color="black", linewidth=1, label="u (управление)"
)
ax4.set_title("Управление", fontsize=9)
ax4.set_xlabel("Время, с", fontsize=8)
ax4.set_ylabel("Значения u", fontsize=8)
ax4.yaxis.set_major_locator(MultipleLocator(0.5))
ax4.xaxis.set_major_locator(MultipleLocator(0.5))
ax4.grid(True, linestyle=":", linewidth=0.5)
ax4.legend(fontsize=9)
ax4.set_xlim(left=0)
ax4.set_ylim(-0.5, np.max(gbest) * 1.1 + 0.01)
ax4.tick_params(labelsize=6)


# ======== Рисуем график сходимости PSO
ax5 = fig.add_subplot(gs[2, 2:4])
ax5.plot(convergence, color="black", linewidth=0.8)
ax5.set_title("Сходимость PSO", fontsize=9)
ax5.set_xlabel("Итерации", fontsize=8)
ax5.set_ylabel("Приспособленность", fontsize=8)
ax5.grid(True, linestyle=":", linewidth=0.4)
ax5.tick_params(labelsize=6)
plt.tight_layout()


# ======== Сохраним графики
# Шаблонное название подпапки для конкретных данных маятника, управления, интервала и pso
new_folder_name = f"pendulum_J{J}_m{m}_h{h}_M{M}_S{S}_N{N}_iter{max_iter}_tau{tau}_tmax{t_max}".replace(
    ".", "-"
)
base_folder = Path("results")
dir = base_folder / f"{new_folder_name}_v1"
# Добавляем версию, если папка уже существует
ver = 1
while dir.exists():
    ver += 1
    dir = base_folder / f"{new_folder_name}_v{ver}"
# Создаем финальную папку
dir.mkdir(parents=True, exist_ok=False)
# Сохраняем общее изображение
image_path = dir / f"summary_time{time_spent:.2f}.png"
plt.savefig(image_path, dpi=300)
print(f"Общий график сохранён в файл: {image_path}")
plt.show()


# ========== Сохраняем каждый график по отдельности

# 1. График φ(t) (угол с управлением)
fig_phi, ax_phi = plt.subplots(figsize=(6, 4))
ax_phi.plot(t, phi, color="black", linewidth=1)
ax_phi.set_title("Динамика φ (угол)", fontsize=10)
ax_phi.set_xlabel("Время, с", fontsize=9)
ax_phi.set_ylabel("φ", fontsize=9)
ax_phi.yaxis.set_major_locator(MultipleLocator(0.2))
# с 1 до 0.5 для основных делений по оси x
ax_phi.xaxis.set_major_locator(MultipleLocator(0.5))
ax_phi.grid(True, linestyle=":", linewidth=0.5)
ax_phi.set_xlim(left=0)
ax_phi.set_ylim(
    min(np.min(phi), np.min(omega)) * 1.1, max(np.max(phi), np.max(omega)) * 1.1
)
ax_phi.tick_params(labelsize=6)
fig_phi.tight_layout()
fig_phi.savefig(dir / "phi.png", dpi=300)
plt.close(fig_phi)

# 2. График ω(t) (угловая скорость)
fig_omega, ax_omega = plt.subplots(figsize=(6, 4))
ax_omega.plot(t, omega, color="gray", linewidth=1)
ax_omega.plot(t, zero_omega, color="slategray", linestyle=":", linewidth=1)
ax_omega.set_title("Динамика ω (скорость)", fontsize=10)
ax_omega.set_xlabel("Время, с", fontsize=9)
ax_omega.set_ylabel("ω", fontsize=9)
ax_omega.grid(True, linestyle=":")
fig_omega.tight_layout()
fig_omega.savefig(dir / "omega.png", dpi=300)
plt.close(fig_omega)

# 3. График управления u(t)
fig_u, ax_u = plt.subplots(figsize=(6, 4))
ax_u.step(u_time, gbest_step, where="post", color="black", linewidth=1)
ax_u.set_title("Управление u(t)", fontsize=10)
ax_u.set_xlabel("Время, с", fontsize=9)
ax_u.set_ylabel("u", fontsize=9)
ax_u.grid(True, linestyle=":")
fig_u.tight_layout()
fig_u.savefig(dir / "u.png", dpi=300)
plt.close(fig_u)

# 4. График сходимости PSO
fig_conv, ax_conv = plt.subplots(figsize=(6, 4))
ax_conv.plot(convergence, color="black", linewidth=0.8)
ax_conv.set_title("Сходимость PSO", fontsize=10)
ax_conv.set_xlabel("Итерации", fontsize=9)
ax_conv.set_ylabel("L", fontsize=9)
ax_conv.grid(True, linestyle=":")
fig_conv.tight_layout()
fig_conv.savefig(dir / "convergence.png", dpi=300)
plt.close(fig_conv)

# 5. График динамики без управления φ(t)
fig_zero, ax_zero = plt.subplots(figsize=(6, 4))
ax_zero.plot(t, zero_phi, color="slategray", linestyle="--", linewidth=0.7)
ax_zero.set_title("Маятник без управления (φ)", fontsize=10)
ax_zero.set_xlabel("Время, с", fontsize=9)
ax_zero.set_ylabel("φ", fontsize=9)
ax_zero.grid(True, linestyle=":")
fig_zero.tight_layout()
fig_zero.savefig(dir / "phi_zero.png", dpi=300)
plt.close(fig_zero)

print(f"Все графики сохранены в папку: {dir}")
