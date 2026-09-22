#!/opt/anaconda3/bin/python3
"""
DE 预测脚本
用法:
  python3 predict.py --thm 5.0 --ph 5.7 --light 520 --catalyst 1.0 --time 180
  python3 predict.py                     # 交互模式
"""

import pickle
import numpy as np
import sys
import os

model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, 'best_model.pkl')

if not os.path.exists(model_path):
    print(f"错误: 模型文件不存在，请先运行 predict_model.py")
    sys.exit(1)

with open(model_path, 'rb') as f:
    saved = pickle.load(f)

model = saved['model']
scaler = saved['scaler']
model_name = saved['model_name']
feature_names = saved['feature_names']

BASE_FEATURES = ['THM', 'pH', 'Light', 'CatalystDosage', 'Time']

def build_features(thm, ph, light, catalyst, time):
    """构建特征向量（与训练时一致）"""
    ph_sq = ph ** 2
    log_time = np.log1p(time)
    thm_inv = 1 / (thm + 0.01)
    return np.array([[thm, ph, light, catalyst, time, ph_sq, log_time, thm_inv]])

def predict(thm, ph, light, catalyst, time):
    X = build_features(thm, ph, light, catalyst, time)

    if model_name == 'Ridge+Poly2':
        pred = model.predict(X)[0]
    else:
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]

    return pred

def print_result(thm, ph, light, catalyst, time, pred):
    print(f"\n{'='*45}")
    print(f"  DE 预测结果")
    print(f"{'='*45}")
    print(f"  THM            = {thm:.2f}")
    print(f"  pH             = {ph:.2f}")
    print(f"  Light          = {light:.0f}")
    print(f"  CatalystDosage = {catalyst:.2f}")
    print(f"  Time           = {time:.0f} min")
    print(f"{'─'*45}")
    print(f"  预测 DE        = {pred:.2f} %")
    print(f"{'='*45}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='预测降解效率 DE')
    parser.add_argument('--thm', type=float, help='初始浓度')
    parser.add_argument('--ph', type=float, help='pH 值')
    parser.add_argument('--light', type=float, help='光照强度')
    parser.add_argument('--catalyst', type=float, help='催化剂用量')
    parser.add_argument('--time', type=float, help='反应时间 (min)')
    parser.add_argument('--interactive', '-i', action='store_true', help='强制交互模式')

    args = parser.parse_args()

    # 命令行模式
    if all(v is not None for v in [args.thm, args.ph, args.light, args.catalyst, args.time]):
        pred = predict(args.thm, args.ph, args.light, args.catalyst, args.time)
        print_result(args.thm, args.ph, args.light, args.catalyst, args.time, pred)
        sys.exit(0)

    # 交互模式
    print("🍅 DE 预测工具 (交互模式)")
    print("输入 q 退出\n")
    print(f"使用模型: {model_name}")
    print(f"特征: {', '.join(feature_names)}\n")

    try:
        while True:
            try:
                thm = input("  THM           > ").strip()
                if thm.lower() == 'q': break
                thm = float(thm)

                ph = input("  pH            > ").strip()
                if ph.lower() == 'q': break
                ph = float(ph)

                light = input("  Light         > ").strip()
                if light.lower() == 'q': break
                light = float(light)

                catalyst = input("  CatalystDosage> ").strip()
                if catalyst.lower() == 'q': break
                catalyst = float(catalyst)

                time_val = input("  Time (min)    > ").strip()
                if time_val.lower() == 'q': break
                time_val = float(time_val)

                pred = predict(thm, ph, light, catalyst, time_val)
                print_result(thm, ph, light, catalyst, time_val, pred)

            except ValueError:
                print("  输入无效，请输入数值\n")
    except (KeyboardInterrupt, EOFError):
        print("\n\n再见！")
        sys.exit(0)
