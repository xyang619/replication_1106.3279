# https://arxiv.org/abs/1106.3279
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main(mu, sigma, A, k, gamma, b, T_minutes, q_max, dt):
    # 将时间从分钟转换为秒
    T = T_minutes * 60  # 时间期限 (秒)
    time_steps = int(T / dt)  # 时间步数

    # 计算alpha, beta, eta
    alpha = (k / 2) * gamma * sigma**2
    beta = k * mu
    eta = A * (1 + gamma / k)**(-(1 + k / gamma))

    # 初始化w_q(t)
    w = np.zeros((q_max + 1, time_steps + 1))
    for q in range(q_max + 1):
        w[q, -1] = np.exp(-k * q * b)  # 终值条件

    # 使用欧拉法求解ODE系统
    for t in range(time_steps - 1, -1, -1):
        for q in range(q_max + 1):
            if q == 0:
                w[q, t] = 1.0  # 边界条件
            else:
                w[q, t] = w[q, t + 1] - dt * ((alpha * q**2 - beta * q) * w[q, t + 1] - eta * w[q - 1, t + 1])

    # 计算最优报价
    delta_a_star = np.zeros((q_max + 1, time_steps + 1))
    for t in range(time_steps + 1):
        for q in range(1, q_max + 1):
            delta_a_star[q, t] = (1 / k) * np.log(w[q, t] / w[q - 1, t]) + (1 / gamma) * np.log(1 + gamma / k)

    # 绘制最优报价
    time = np.linspace(0, T_minutes, time_steps + 1)  # 转换为分钟
    for q in range(1, q_max + 1):
        plt.plot(time, delta_a_star[q, :], label=f'q={q}')

    plt.xlabel('Time (minutes)')
    plt.ylabel('Optimal Quote (ticks)')
    plt.title('Optimal Quotes for Different Inventory Levels')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="Optimal Portfolio Liquidation with Limit Orders")
    parser.add_argument("--mu", type=float, default=0.0, help="Drift rate (Tick/s)[0]")
    parser.add_argument("--sigma", type=float, default=0.3, help="Volatility (Tick/s^0.5)[0.3]")
    parser.add_argument("--A", type=float, default=0.1, help="Intensity scale parameter (1/s)[0.1]")
    parser.add_argument("--k", type=float, default=0.3, help="Intensity shape parameter (1/Tick)[0.3]")
    parser.add_argument("--gamma", type=float, default=0.05, help="Risk aversion coefficient (1/Tick)[0.05]")
    parser.add_argument("--b", type=float, default=3.0, help="Liquidation cost (Tick)[3]")
    parser.add_argument("--T_minutes", type=float, default=5.0, help="Time horizon (minutes)[5]")
    parser.add_argument("--q_max", type=int, default=6, help="Maximum inventory[6]")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step (seconds)[0.1]")

    args = parser.parse_args()

    # 调用主函数
    main(args.mu, args.sigma, args.A, args.k, args.gamma, args.b, args.T_minutes, args.q_max, args.dt)

# Generate Figure 1 using default configuration
# Generate Figure 2 by change --T_minutes 120
