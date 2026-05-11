import numpy as np
import pandas as pd

class RiskMetrics:
    """
    投资组合风险与绩效指标计算工具
    对应面试考点：夏普比率计算、最大回撤算法、年化逻辑
    """
    
    @staticmethod
    def calculate_all(returns_series, rf=0.0):
        """
        输入: 
            returns_series: pd.Series, 每日收益率序列 (样本外)
            rf: float, 无风险利率 (默认为 0)
        返回: 
            dict: 包含年化收益、波动率、夏普比率、最大回撤、卡玛比率
        """
        # 确保数据没有空值
        res = returns_series.dropna()
        if len(res) == 0:
            return {}

        # 1. 计算累计净值 (Cumulative Net Value)
        cum_net_value = (1 + res).cumprod()
        
        # 2. 年化收益率 (Annualized Return)
        # 假设一年 252 个交易日
        total_return = cum_net_value.iloc[-1] - 1
        ann_return = res.mean() * 252
        
        # 3. 年化波动率 (Annualized Volatility)
        ann_vol = res.std() * np.sqrt(252)
        
        # 4. 夏普比率 (Sharpe Ratio)
        # 面试重点：夏普比率衡量每单位风险带来的超额收益
        sharpe_ratio = (ann_return - rf) / ann_vol if ann_vol != 0 else 0
        
        # 5. 最大回撤 (Maximum Drawdown)
        # 算法逻辑：当前净值相对于历史最高点的最大跌幅
        historical_max = cum_net_value.cummax()
        drawdowns = (cum_net_value - historical_max) / historical_max
        max_drawdown = drawdowns.min()
        
        # 6. 卡玛比率 (Calmar Ratio)
        # 很多私募更看重卡玛比率：年化收益 / |最大回撤|
        calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            "Annualized Return": ann_return,
            "Annualized Vol": ann_vol,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Calmar Ratio": calmar_ratio,
            "Total Return": total_return
        }

    @staticmethod
    def get_equity_curve(returns_series):
        """返回累计净值曲线，方便绘图"""
        return (1 + returns_series).cumprod()