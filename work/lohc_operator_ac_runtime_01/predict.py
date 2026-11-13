# ===============================================
# predict_delta_H_auto.py — 支持杂原子的焓变预测代码
# 核心修改：增强杂原子SMILES解析、兼容杂原子图特征、补充调试信息
# ===============================================
import os
import copy
from collections import OrderedDict
from multiprocessing import current_process
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

# 关闭RDKit冗余日志（避免杂原子解析时的警告刷屏）
RDLogger.DisableLog('rdApp.*')

# --------------------------
# 1. 导入GNN2.py的核心组件
# --------------------------
from GNN2 import (
    CONFIG, SEED, smiles_to_graph, SiameseGIN, collate_fn
)

# --------------------------
# 2. 核心配置（仅需保证训练集路径正确！）
# --------------------------
ROOT = Path(__file__).resolve().parent

PREDICT_CFG = {
    "model_weights_dir": CONFIG["save_dir"],  # fold权重目录（无需改）
    "train_data_path": str(ROOT / "QM9-LOHC_new_molecules.csv"),
    "device": CONFIG["device"],               # 设备（无需改）
    "num_folds": 5,                           # 5折CV（无需改）
    "batch_size": CONFIG.get("batch_size", 32),
    "num_workers": CONFIG.get("num_workers", 0),
    # ===================== 新增：支持的杂原子列表 =====================
    "supported_atoms": ['C', 'H', 'O', 'N', 'F', 'Cl', 'Br', 'S', 'P'],  # 扩展杂原子
}

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

SMILES_PREPROCESS_CACHE = {}
GRAPH_CACHE = OrderedDict()
GRAPH_CACHE_MAXSIZE = int(os.environ.get('LOHC_GRAPH_CACHE_MAXSIZE', '20000'))

# --------------------------
# 3. 自动加载训练集并计算反归一化参数（无修改）
# --------------------------
def load_train_scaler_params():
    """自动读取训练集，计算delta_H的均值和标准差"""
    if not os.path.exists(PREDICT_CFG["train_data_path"]):
        raise FileNotFoundError(
            f"训练集文件未找到：{PREDICT_CFG['train_data_path']}\n"
            "请检查路径是否正确，例如：是否有中文/空格、路径分隔符是否正确（Windows用\\，Linux用/）"
        )
    
    # 读取训练集并过滤有效delta_H
    df = pd.read_csv(PREDICT_CFG["train_data_path"])
    if "delta_H" not in df.columns:
        raise ValueError("训练集缺少'delta_H'列！请确认训练集格式和GNN2一致")
    
    delta_H = df["delta_H"].dropna().values
    if len(delta_H) == 0:
        raise ValueError("训练集'delta_H'列无有效数据！")
    
    # 计算均值和标准差（和GNN2训练时完全一致）
    mean = np.mean(delta_H)
    std = np.std(delta_H)
    if current_process().name == "MainProcess":
        print(f"自动读取训练集成功！")
        print(f"   delta_H均值：{mean:.4f} | 标准差：{std:.4f}")
    return mean, std

DELTA_H_MEAN = None
DELTA_H_STD = None

def _ensure_scaler_params():
    global DELTA_H_MEAN, DELTA_H_STD
    if DELTA_H_MEAN is None or DELTA_H_STD is None:
        DELTA_H_MEAN, DELTA_H_STD = load_train_scaler_params()

# --------------------------
# 4. 反归一化函数（无修改）
# --------------------------
def denormalize(pred_scaled):
    """反归一化：自动使用训练集参数"""
    _ensure_scaler_params()
    return pred_scaled * DELTA_H_STD + DELTA_H_MEAN

# --------------------------
# ===================== 关键修改1：增强杂原子SMILES解析 =====================
# 新增：鲁棒的SMILES预处理函数（处理杂原子格式、原子价等问题）
# --------------------------
def preprocess_smiles(smiles, mol_type="饱和"):
    """
    预处理SMILES，保证杂原子能正确解析：
    1. 标准化SMILES格式（修复杂原子的原子价/环结构）
    2. 检查是否包含不支持的原子
    3. 返回标准化后的SMILES（失败返回None）
    """
    cache_key = (mol_type, smiles)
    if cache_key in SMILES_PREPROCESS_CACHE:
        return SMILES_PREPROCESS_CACHE[cache_key]
    # 与GA一致的SMILES处理逻辑
    if not isinstance(smiles, str) or len(smiles) == 0:
        print(f" {mol_type}SMILES格式错误：{smiles}（非字符串/空值）")
        SMILES_PREPROCESS_CACHE[cache_key] = None
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f" {mol_type}SMILES解析失败：{smiles}（原始格式非法）")
            SMILES_PREPROCESS_CACHE[cache_key] = None
            return None
        mol = Chem.AddHs(mol)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        normalized_smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        SMILES_PREPROCESS_CACHE[cache_key] = normalized_smiles
        return normalized_smiles
    except Exception as e:
        print(f"{mol_type}SMILES{smiles}预处理失败：{e}")
        SMILES_PREPROCESS_CACHE[cache_key] = None
        return None

def _smiles_to_graph_cached(smiles):
    if smiles in GRAPH_CACHE:
        graph = GRAPH_CACHE.pop(smiles)
        GRAPH_CACHE[smiles] = graph
        return copy.deepcopy(graph)
    graph = smiles_to_graph(smiles)
    if graph is None:
        GRAPH_CACHE[smiles] = None
        while len(GRAPH_CACHE) > GRAPH_CACHE_MAXSIZE:
            GRAPH_CACHE.popitem(last=False)
        return None
    GRAPH_CACHE[smiles] = copy.deepcopy(graph)
    while len(GRAPH_CACHE) > GRAPH_CACHE_MAXSIZE:
        GRAPH_CACHE.popitem(last=False)
    return copy.deepcopy(graph)

# --------------------------
# 5. 加载GNN2模型（无修改）
# --------------------------
def load_gnn2_models():
    """加载num_folds个fold权重，缺失则报错"""
    models, missing = [], []
    dummy_graph = smiles_to_graph("C(F)")
    in_features = dummy_graph.x.size(1)
    for fold in range(1, PREDICT_CFG["num_folds"] + 1):
        weight_path = os.path.join(PREDICT_CFG["model_weights_dir"], f"fold{fold}_best.pth")
        if not os.path.exists(weight_path):
            missing.append(weight_path)
            continue
        model = SiameseGIN(
            in_features=in_features,
            h1=CONFIG["h1"], h2=CONFIG["h2"], h3=CONFIG["h3"],
            dropout=CONFIG["dropout"]
        ).to(PREDICT_CFG["device"])
        model.load_state_dict(torch.load(weight_path, map_location=PREDICT_CFG["device"]))
        # A fixed point prediction keeps the same pair comparable across arms.
        model.eval()
        models.append(model)
    if missing or len(models) != PREDICT_CFG["num_folds"]:
        raise FileNotFoundError(f"缺失fold权重：{missing}")
    print(f"成功加载{len(models)}折GNN2模型，用于平均预测")
    return models

# --------------------------
# 6. 核心预测函数（修改杂原子处理逻辑）
# --------------------------
def _predict_delta_H_ensemble(sat_smiles_list, unsat_smiles_list):
    """
    预测焓变 ensemble 均值和模型间标准差（支持杂原子）。
    """
    # 加载模型（仅加载一次）
    if not hasattr(predict_delta_H, "models"):
        predict_delta_H.models = load_gnn2_models()
    
    # 处理SMILES转图（新增杂原子预处理）
    dataset = []
    invalid_indices = set()
    for idx, (sat_smi, unsat_smi) in enumerate(zip(sat_smiles_list, unsat_smiles_list)):
        # ===================== 关键修改2：预处理杂原子SMILES =====================
        # 1. 预处理饱和SMILES（处理杂原子）
        sat_smi_processed = preprocess_smiles(sat_smi, mol_type="饱和")
        if sat_smi_processed is None:
            invalid_indices.add(idx)
            continue
        # 2. 预处理不饱和SMILES（处理杂原子）
        unsat_smi_processed = preprocess_smiles(unsat_smi, mol_type="不饱和")
        if unsat_smi_processed is None:
            invalid_indices.add(idx)
            continue
        
        # 3. 转图（使用预处理后的SMILES，保证杂原子能正确生成图特征）
        sat_graph = _smiles_to_graph_cached(sat_smi_processed)
        unsat_graph = _smiles_to_graph_cached(unsat_smi_processed)
        if sat_graph is None or unsat_graph is None:
            print(f"SMILES转图失败：饱和={sat_smi_processed} | 不饱和={unsat_smi_processed}")
            invalid_indices.add(idx)
            continue
        
        dataset.append((sat_graph, unsat_graph, 0.0))
    
    # 无有效数据返回NaN
    if len(dataset) == 0:
        print("无有效SMILES数据用于预测")
        nan_list = [np.nan] * len(sat_smiles_list)
        return nan_list, nan_list.copy()
    
    # 构建DataLoader
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=PREDICT_CFG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=int(PREDICT_CFG.get("num_workers", 0))
    )
    loader = DataLoader(**loader_kwargs)
    
    # 集成预测
    all_fold_preds = []
    with torch.no_grad():
        for model in predict_delta_H.models:
            fold_preds = []
            for sat_batch, unsat_batch, _ in loader:
                sat_batch = sat_batch.to(PREDICT_CFG["device"])
                unsat_batch = unsat_batch.to(PREDICT_CFG["device"])
                pred = model(sat_batch, unsat_batch)
                fold_preds.extend(pred.cpu().numpy().tolist())
            all_fold_preds.append(fold_preds)
    # 显式堆叠后按fold维计算均值和模型间标准差。
    # 标准差作为轻量 Bayesian/ensemble uncertainty 传递给 GA。
    pred_scaled = np.stack(all_fold_preds, axis=0)
    avg_pred_scaled = np.mean(pred_scaled, axis=0)
    std_pred_scaled = np.std(pred_scaled, axis=0, ddof=0)
    avg_pred_denorm = denormalize(avg_pred_scaled)
    std_pred_denorm = std_pred_scaled * DELTA_H_STD
    
    # 组装结果
    final_mean = []
    final_std = []
    valid_idx = 0
    for idx in range(len(sat_smiles_list)):
        if idx in invalid_indices:
            final_mean.append(np.nan)
            final_std.append(np.nan)
        else:
            final_mean.append(float(avg_pred_denorm[valid_idx]))
            final_std.append(float(std_pred_denorm[valid_idx]))
            valid_idx += 1
    
    return final_mean, final_std


def predict_delta_H(sat_smiles_list, unsat_smiles_list):
    """
    预测焓变均值，保持旧 GA 调用接口不变。
    """
    mean, _ = _predict_delta_H_ensemble(sat_smiles_list, unsat_smiles_list)
    return mean


def predict_delta_H_with_uncertainty(sat_smiles_list, unsat_smiles_list):
    """
    返回 5-fold ensemble 的焓变均值和模型间标准差。
    标准差用于 GA 的 uncertainty-aware / Bayesian fitness penalty。
    """
    return _predict_delta_H_ensemble(sat_smiles_list, unsat_smiles_list)

# --------------------------
# 7. 测试运行（修改测试用例为含杂原子的SMILES）
# --------------------------
if __name__ == "__main__":
    # ========== 修改：测试含杂原子的SMILES对 ==========
    # 示例：含F的饱和/不饱和对（环己烷-F vs 苯-F）
    test_sat_smiles = ["FC1CNCCC1CC"]       # 含F的饱和SMILES
    test_unsat_smiles = ["FC1=CN=CC=C1C#C"]     # 含F的不饱和SMILES
    # ========== 结束修改 ==========

    # 执行预测
    preds = predict_delta_H(test_sat_smiles, test_unsat_smiles)
    
    # 打印结果
    print("\n 焓变预测结果（单位：kJ/mol）：")
    for i, (sat, unsat, pred) in enumerate(zip(test_sat_smiles, test_unsat_smiles, preds)):
        if np.isnan(pred):
            print(f"序号{i+1} | 饱和SMILES：{sat} | 不饱和SMILES：{unsat} | 结果：无效SMILES（含不支持原子/格式错误）")
        else:
            print(f"序号{i+1} | 饱和SMILES：{sat} | 不饱和SMILES：{unsat} | 预测焓变：{pred:.3f}")
