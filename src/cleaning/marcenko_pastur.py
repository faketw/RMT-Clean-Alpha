import numpy as np

class MarcenkoPastur:
    """
    Marčenko-Pastur 理论分布工具
    对应论文: Section 2.2.3 The Marčenko-Pastur equation
    """
    @staticmethod
    def get_theoretical_edge(sigma_sq, Q):
        """
        计算 MP 分布的理论上下界 lambda_+ 和 lambda_-
        Q = T / N (样本量与维度比)
        """
        inv_q = 1.0 / Q
        l_plus = sigma_sq * (1 + np.sqrt(inv_q))**2
        l_minus = sigma_sq * (1 - np.sqrt(inv_q))**2
        return l_minus, l_plus

    @staticmethod
    def pdf(x, sigma_sq, Q):
        """
        MP 分布的概率密度函数 (PDF)
        用于在 GitHub README 中绘制理论曲线与实验数据的对比
        """
        l_min, l_max = MarcenkoPastur.get_theoretical_edge(sigma_sq, Q)
        # 仅在边界内计算
        res = (Q / (2 * np.pi * sigma_sq * x)) * np.sqrt(np.maximum(0, (l_max - x) * (x - l_min)))
        return np.nan_to_num(res)

    @staticmethod
    def estimate_sigma_sq(evals, Q):
        """
        自适应估计噪声方差 (对应论文 Section 3.2.1)
        假设落入理论边界内的特征值属于噪声。
        """
        # 初始估计：假设 sigma_sq 为 1
        l_min, l_max = MarcenkoPastur.get_theoretical_edge(1.0, Q)
        noise_evals = evals[evals <= l_max]
        if len(noise_evals) == 0:
            return np.mean(evals) # 退化处理
        return np.mean(noise_evals)