import numpy as np

class OptimalRIE:
    """
    最佳旋转不变估计量 (RIE)
    对应论文: Section 5.4 The "Optimal" RIE
    这是处理 N 接近 T 时（高 Q 值）最稳健的方法。
    """
    def __init__(self, corr_matrix, Q):
        self.corr_matrix = corr_matrix
        self.Q = Q # Q = T/N

    def clean(self):
        """
        应用最佳非线性收缩公式修正特征值偏误
        """
        evals, evecs = np.linalg.eigh(self.corr_matrix)
        q_inv = 1.0 / self.Q
        
        # 构造 Stieltjes 变换的数值近似
        # 这是物理学处理连续能谱的常用手段
        cleaned_evals = []
        for l in evals:
            # 这里的公式对应论文中关于 xi_rie(lambda) 的推导
            # 为了数值稳定性，引入一个小扰动 eta
            eta = evals.max() * (len(evals)**-0.5)
            z = l - 1j * eta
            
            # 计算样本解析函数 (Resolvent)
            g_z = np.mean(1.0 / (z - evals))
            
            # RIE 核心修正公式: 修正特征值的偏差
            # 论文公式 (5.40) 的变体
            xi = l / abs(1 - q_inv + q_inv * l * g_z)**2
            cleaned_evals.append(xi)
            
        cleaned_evals = np.array(cleaned_evals)
        
        # 重构
        corr_rie = evecs @ np.diag(cleaned_evals) @ evecs.T
        
        # 归一化
        d = np.diag(corr_rie)
        std = np.sqrt(d)
        corr_rie = corr_rie / np.outer(std, std)
        
        return corr_rie, cleaned_evals