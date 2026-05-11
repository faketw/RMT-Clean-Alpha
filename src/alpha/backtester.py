import pandas as pd
import numpy as np
from src.cleaning.optimal_rie import OptimalRIE

class OutOfSampleEngine:
    def __init__(self, returns_df, window_size=252):
        self.returns = returns_df
        self.window_size = window_size
        self.N = returns_df.shape[1] # 提取资产数量

    def _compute_weights(self, corr, stds):
        """核心权重计算：包含稳定性约束"""
        # 1. 构建协方差矩阵 (Sigma = D * Corr * D)
        cov = np.diag(stds) @ corr @ np.diag(stds)
        
        # 2. 正则化 (Tikhonov)
        cov += np.eye(len(corr)) * 1e-6
        
        # 3. 计算最小方差权重
        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(len(corr))
        w = inv_cov @ ones
        w /= np.sum(w) 
        
        # 4. 极端权重限制 (Clipping)
        # 这一步防止了单日 -105% 的惨剧
        w = np.clip(w, -0.05, 0.10)
        w /= np.sum(w) # 重新归一化
        
        return w

    def run_rolling_cleaning(self, method='rie', fee=0.002):
        T, N = self.returns.shape
        out_of_sample_results = []
        weight_records = []
        actual_dates = self.returns.index[self.window_size + 1 : T]
        last_w = np.zeros(N) # 初始仓位为0
        
        # 注意：这里循环到 T-1 确保 t+1 不会越界
        for t in range(self.window_size, T - 1):
            in_sample_data = self.returns.iloc[t - self.window_size : t]
            stds = in_sample_data.std().values # 提取波动率
            corr_orig = in_sample_data.corr().values
            Q = self.window_size / N
            
            # 选择去噪方法
            if method == 'sample':
                corr_clean = corr_orig
            elif method == 'rie':
                cleaner = OptimalRIE(corr_orig, Q)
                corr_clean, _ = cleaner.clean()
            else:
                corr_clean = corr_orig

            # 带约束的函数
            w = self._compute_weights(corr_clean, stds)
            turnover = np.sum(np.abs(w - last_w))
            cost = turnover * fee
            
            # 获取下一日真实收益
            next_day_ret = self.returns.iloc[t + 1].values
            gross_ret = np.dot(w, next_day_ret)
            net_ret = gross_ret - cost
            
            out_of_sample_results.append(net_ret)
            weight_records.append(w)
            last_w = w
        
        res_series = pd.Series(out_of_sample_results, index=actual_dates)
        res_weights = pd.DataFrame(weight_records, index=actual_dates)
        
        return res_series, res_weights