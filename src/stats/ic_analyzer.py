import numpy as np
import pandas as pd
from scipy import stats

class ICAnalyzer:
    """计算信息系数(IC)与显著性，体现统计建模能力"""
    
    @staticmethod
    def analyze(weights_df, next_returns_df):
        """
        核心修复逻辑：
        1. 自动提取两个 DataFrame 的公共日期（Intersection）
        2. 确保在计算前剔除包含 NaN 的行
        """
        # --- 步骤 1: 强制对齐日期索引 ---
        # 只有当两个表都有数据的日期，才能计算 IC
        common_dates = weights_df.index.intersection(next_returns_df.index)
        
        if len(common_dates) == 0:
            print("Warning: No common dates found between weights and returns. Check your indices!")
            return {"Mean Rank IC": 0, "IC t-stat": 0, "p-value": 1.0}

        # 仅保留交集部分的数据
        w_aligned = weights_df.loc[common_dates]
        r_aligned = next_returns_df.loc[common_dates]
        
        ics = []
        for t in common_dates:
            w = w_aligned.loc[t].values
            r = r_aligned.loc[t].values
            
            # --- 步骤 2: 鲁棒性检查 ---
            # 检查是否有 NaN，或权重是否完全一致（std=0 会导致相关系数为 nan）
            if np.isnan(w).any() or np.isnan(r).any() or np.std(w) < 1e-10:
                ics.append(0)
                continue
                
            # 使用 Rank IC (Spearman)
            ric, _ = stats.spearmanr(w, r)
            
            # 处理 spearmanr 返回 nan 的特殊情况
            if np.isnan(ric):
                ics.append(0)
            else:
                ics.append(ric)
            
        ic_series = pd.Series(ics, index=common_dates)
        
        # --- 步骤 3: 统计量计算 ---
        if ic_series.empty or (len(ic_series) < 2) or (ic_series.std() == 0):
            return {"Mean Rank IC": 0, "IC t-stat": 0, "p-value": 1.0}
            
        mean_ic = ic_series.mean()
        # t-stat = mean / standard_error
        t_stat = mean_ic / (ic_series.std() / np.sqrt(len(ic_series)))
        # 双侧检验 p-value
        p_val = stats.t.sf(np.abs(t_stat), len(ic_series)-1) * 2
        
        return {
            "Mean Rank IC": mean_ic, 
            "IC t-stat": t_stat, 
            "p-value": p_val,
            "Count": len(ic_series)
        }