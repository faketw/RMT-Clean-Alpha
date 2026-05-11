from src.alpha.backtester import OutOfSampleEngine
from src.stats.ic_analyzer import ICAnalyzer
from src.stats.risk_metrics import RiskMetrics # 假设已写好
import numpy as np
import pandas as pd

def main():
    # 1. 加载数据
    df = pd.read_csv('data/market_evolution_tracks.csv', index_index=0)
    
    # 2. 启动严格隔离的滚动回测
    engine = OutOfSampleEngine(df, window_size=252)
    oos_returns, oos_weights = engine.run_rolling_cleaning()
    
    # 3. 统计分析
    # 计算 IC 显著性
    next_returns = df.iloc[253:] # 对应样本外时段
    ic_report = ICAnalyzer.analyze(oos_weights, next_returns)
    
    # 4. 输出结论
    print("=== 样本外统计报告 (数据已严格隔离) ===")
    print(f"IC t-stat: {ic_report['IC t-stat']:.2f}")
    print(f"P-Value: {ic_report['p-value']:.4f}")
    # 若 t-stat > 2，说明去噪真实有效！

if __name__ == "__main__":
    main()