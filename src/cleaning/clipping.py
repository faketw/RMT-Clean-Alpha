import numpy as np

class EigenvalueClipper:
    """
    特征值裁剪去噪器 (Constant Residual Hull)
    对应论文: Section 7.2.2 Eigenvalue Power Mapping / Clipping
    """
    def __init__(self, corr_matrix, Q):
        self.corr_matrix = corr_matrix
        self.Q = Q
        self.evals, self.evecs = self._decompose()

    def _decompose(self):
        evals, evecs = np.linalg.eigh(self.corr_matrix)
        idx = evals.argsort()[::-1] # 降序
        return evals[idx], evecs[:, idx]

    def clean(self, sigma_sq=None):
        """
        执行清洗：保留离群点，将噪声特征值设为均值
        """
        from .marcenko_pastur import MarcenkoPastur
        
        if sigma_sq is None:
            sigma_sq = MarcenkoPastur.estimate_sigma_sq(self.evals, self.Q)
            
        _, l_plus = MarcenkoPastur.get_theoretical_edge(sigma_sq, self.Q)
        
        # 确定信号与噪声的掩码
        signal_mask = self.evals > l_plus
        
        # 计算噪声部分的均值（能量守恒）
        cleaned_evals = self.evals.copy()
        avg_noise = np.mean(self.evals[~signal_mask])
        cleaned_evals[~signal_mask] = avg_noise
        
        # 重构矩阵: C = V * Lambda * V^T
        corr_cleaned = self.evecs @ np.diag(cleaned_evals) @ self.evecs.T
        
        # 物理归一化：确保相关矩阵对角线为 1
        d = np.diag(corr_cleaned)
        std = np.sqrt(d)
        corr_cleaned = corr_cleaned / np.outer(std, std)
        
        return corr_cleaned, l_plus, sigma_sq