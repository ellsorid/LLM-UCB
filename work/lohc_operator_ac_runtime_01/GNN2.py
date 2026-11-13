# ===============================================
# GNN2.py — Siamese GIN with 5-Fold CV + EMA
# Updated: Added Train metrics and Loss vs Epoch plot
# ===============================================

import os, copy, random
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader  # FIXED IMPORT
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# --------------------------
# Reproducibility
# --------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

NUM_CPU_THREADS = 1
torch.set_num_threads(NUM_CPU_THREADS)
if hasattr(torch, 'set_num_interop_threads'):
    torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parent

CONFIG = {
    'data_path': str(ROOT / "QM9-LOHC_new_molecules.csv"),
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'dropout': 0.1,
    'train_split': 0.8,
    'num_workers': 0,
    'save_dir': str(ROOT / "model_checkpoints"),
    'device': 'cpu',
    'h1': 256, 'h2': 256, 'h3': 256,
    'warmup_epochs': 10
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)
RDLogger.DisableLog('rdApp.warning')

# --------------------------
# SMILES -> Graph Utilities
# --------------------------
def atom_hybridization_onehot(atom):
    mp = {
        Chem.HybridizationType.S: 1,
        Chem.HybridizationType.SP: 1,
        Chem.HybridizationType.SP2: 2,
        Chem.HybridizationType.SP3: 3,
        Chem.HybridizationType.SP3D: 4,
        Chem.HybridizationType.SP3D2: 5
    }
    return mp.get(atom.GetHybridization(), 0) / 5.0

def compute_rdkit_descriptors(mol):
    try:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        return [mw/500.0, (logp+5.0)/10.0, tpsa/200.0]
    except:
        return [0.0, 0.0, 0.0]

def smiles_to_graph(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None or mol.GetNumAtoms()==0: return None

        atom_features = []
        for atom in mol.GetAtoms():
            at_num = atom.GetAtomicNum()/100.0
            formal_charge = atom.GetFormalCharge()
            total_H = atom.GetTotalNumHs()/4.0
            is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
            hybrid = atom_hybridization_onehot(atom)
            degree = atom.GetDegree()/6.0
            exp_val = getattr(atom, "GetExplicitValence", lambda:0)()/6.0
            imp_val = getattr(atom, "GetImplicitValence", lambda:0)()/6.0
            total_val = getattr(atom, "GetTotalValence", lambda:0)()/12.0
            mass = getattr(atom, "GetMass", lambda: atom.GetAtomicNum())()/300.0
            is_chiral = 1.0 if atom.HasProp('_CIPCode') else 0.0
            atom_features.append([at_num, formal_charge, total_H, is_aromatic, hybrid,
                                  degree, exp_val, imp_val, total_val, mass, is_chiral])
        x = torch.tensor(atom_features, dtype=torch.float)

        edge_index, edge_attr = [], []
        for bond in mol.GetBonds():
            i,j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = float(bond.GetBondTypeAsDouble())/3.0
            conj = 1.0 if bond.GetIsConjugated() else 0.0
            ring = 1.0 if bond.IsInRing() else 0.0
            edge_index += [[i,j],[j,i]]
            edge_attr += [[bt,conj,ring],[bt,conj,ring]]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.empty((2,0), dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float) if edge_attr else torch.empty((0,3), dtype=torch.float)

        mol_feat = torch.tensor(compute_rdkit_descriptors(mol), dtype=torch.float).unsqueeze(0)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=x.size(0), mol_feat=mol_feat)
    except:
        return None

# --------------------------
# Dataset Preparation
# --------------------------
def prepare_dataset(df, sat_col='sat_SMILE', unsat_col='unsat_SMILE', target_col='delta_H'):
    dataset, invalid = [], 0
    for _, row in df.iterrows():
        s1, s2 = row.get(sat_col), row.get(unsat_col)
        if not isinstance(s1,str) or not isinstance(s2,str):
            invalid+=1; continue
        g1, g2 = smiles_to_graph(s1), smiles_to_graph(s2)
        if g1 is None or g2 is None: invalid+=1; continue
        y = row.get(target_col)
        if y is None or (isinstance(y,float) and np.isnan(y)): invalid+=1; continue
        dataset.append((g1,g2,float(y)))
    print(f"Valid pairs: {len(dataset)}, Invalid pairs: {invalid}")
    return dataset

def collate_fn(batch):
    sat_graphs, unsat_graphs, ys = zip(*batch)
    return Batch.from_data_list(sat_graphs), Batch.from_data_list(unsat_graphs), torch.tensor(ys, dtype=torch.float)

# ========================== 推理辅助与统一预测 ==========================
def _fit_scaler(train_path=CONFIG['data_path']):
    df = pd.read_csv(train_path)
    if 'delta_H' not in df.columns:
        raise ValueError("训练集缺少 'delta_H' 列")
    delta_H = df['delta_H'].dropna().values.reshape(-1, 1)
    if delta_H.size == 0:
        raise ValueError("'delta_H' 列无有效数据")
    scaler = StandardScaler().fit(delta_H)
    return scaler

def _load_infer_models(num_folds=5, device=CONFIG['device']):
    models, missing = [], []
    dummy = smiles_to_graph("C")
    in_features = dummy.x.size(1)
    for fold in range(1, num_folds + 1):
        w = os.path.join(CONFIG['save_dir'], f"fold{fold}_best.pth")
        if not os.path.exists(w):
            missing.append(w); continue
        model = SiameseGIN(in_features, CONFIG['h1'], CONFIG['h2'], CONFIG['h3'], dropout=CONFIG['dropout']).to(device)
        model.load_state_dict(torch.load(w, map_location=device))
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()  # 与验证阶段一致：激活dropout
        models.append(model)
    if missing or len(models) != num_folds:
        raise FileNotFoundError(f"缺失fold权重：{missing}")
    return models

def predict_delta_H_gnn2(sat_smiles_list, unsat_smiles_list, train_path=None, device=None, batch_size=None, num_folds=None, targets=None, fig_path=None):
    device = device or CONFIG['device']
    batch_size = batch_size or CONFIG['batch_size']
    num_folds = num_folds or 5
    train_path = train_path or CONFIG['data_path']

    scaler = _fit_scaler(train_path)
    mean, std = scaler.mean_[0], scaler.scale_[0]

    if not hasattr(predict_delta_H_gnn2, "_models") or predict_delta_H_gnn2._device != device or predict_delta_H_gnn2._num_folds != num_folds:
        predict_delta_H_gnn2._models = _load_infer_models(num_folds=num_folds, device=device)
        predict_delta_H_gnn2._device = device
        predict_delta_H_gnn2._num_folds = num_folds

    dataset, invalid_idx = [], []
    for idx, (s_sat, s_unsat) in enumerate(zip(sat_smiles_list, unsat_smiles_list)):
        g1, g2 = smiles_to_graph(s_sat), smiles_to_graph(s_unsat)
        if g1 is None or g2 is None:
            invalid_idx.append(idx); continue
        dataset.append((g1, g2, 0.0))
    if not dataset:
        return [np.nan] * len(sat_smiles_list)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    all_fold_preds = []
    with torch.no_grad():
        for model in predict_delta_H_gnn2._models:
            fold_preds = []
            for sat_batch, unsat_batch, _ in loader:
                sat_batch, unsat_batch = sat_batch.to(device), unsat_batch.to(device)
                out = model(sat_batch, unsat_batch)
                fold_preds.extend(out.cpu().numpy().tolist())
            all_fold_preds.append(fold_preds)

    avg_scaled = np.mean(np.stack(all_fold_preds, axis=0), axis=0)
    avg_denorm = avg_scaled * std + mean

    results, v = [], 0
    for i in range(len(sat_smiles_list)):
        if i in invalid_idx:
            results.append(np.nan)
        else:
            results.append(avg_denorm[v]); v += 1

    if targets is not None and fig_path:
        targets = np.array(targets, dtype=float)
        pairs = [(p, t) for p, t in zip(results, targets) if not np.isnan(p)]
        if pairs:
            preds_valid, targets_valid = zip(*pairs)
            plt.figure(figsize=(6,6))
            plt.scatter(targets_valid, preds_valid, alpha=0.6, s=12)
            lo, hi = min(targets_valid), max(targets_valid)
            plt.plot([lo, hi], [lo, hi], '--', color='red')
            plt.xlabel("True δH (kJ/mol)")
            plt.ylabel("Predicted δH (kJ/mol)")
            plt.title("Inference vs Target (GNN2 推理)")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
    return results

# --------------------------
# Model
# --------------------------
def make_node_mlp(h): return nn.Sequential(nn.Linear(h,h), nn.ReLU(), nn.Linear(h,h))

class SiameseGIN(nn.Module):
    def __init__(self, in_features, h1,h2,h3, dropout=0.1, edge_dim=3, mol_desc_dim=3):
        super().__init__()
        assert h1==h2==h3
        h=h1
        self.h=h
        self.node_encoder = nn.Linear(in_features,h)
        self.edge_encoder = nn.Linear(edge_dim,h)
        self.conv1 = GINEConv(make_node_mlp(h)); self.bn1=nn.LayerNorm(h)
        self.conv2 = GINEConv(make_node_mlp(h)); self.bn2=nn.LayerNorm(h)
        self.conv3 = GINEConv(make_node_mlp(h)); self.bn3=nn.LayerNorm(h)
        pooled_dim = h*2 + mol_desc_dim
        self.fc1 = nn.Linear(pooled_dim*2, pooled_dim)
        self.fc2 = nn.Linear(pooled_dim, pooled_dim//2)
        self.fc3 = nn.Linear(pooled_dim//2,1)
        self.dropout=nn.Dropout(dropout)

    def encode(self,batch:Batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, getattr(batch,'edge_attr',None), batch.batch
        x = self.node_encoder(x)
        edge_attr_trans = self.edge_encoder(edge_attr) if edge_attr is not None and edge_attr.numel()>0 else None

        x1 = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr_trans))); x1=self.dropout(x1)
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index, edge_attr_trans))); x2=self.dropout(x2)
        x = x1 + x2
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr_trans)))

        mean_pool = global_mean_pool(x, batch_idx) #Pooling is how the GNN “summarizes” a molecule — it compresses all atom-wise knowledge into one fixed-length molecular fingerprint for downstream prediction.
        max_pool = global_max_pool(x, batch_idx)
        g = torch.cat([mean_pool,max_pool],dim=1)

        if hasattr(batch,'mol_feat'):
            mol_feat = batch.mol_feat.to(g.device)
            if mol_feat.dim()==3 and mol_feat.size(1)==1: mol_feat = mol_feat.squeeze(1)
            g = torch.cat([g,mol_feat],dim=1)
        else: g = torch.cat([g,torch.zeros(g.size(0),3,device=g.device)],dim=1)
        return g

    def forward(self, sat_batch, unsat_batch):
        sat_emb = self.encode(sat_batch)
        unsat_emb = self.encode(unsat_batch)
        combined = torch.cat([sat_emb, unsat_emb], dim=1)
        x = F.relu(self.fc1(combined)); x=self.dropout(x)
        x = F.relu(self.fc2(x)); x=self.dropout(x)
        return self.fc3(x).squeeze(-1)
    def predict_deltaH(self, sat_batch, unsat_batch):
            """Predict the deltaH value for a given pair of saturated and unsaturated molecules."""
            self.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # Forward pass to compute deltaH (the change in enthalpy)
                deltaH = self.forward(sat_batch, unsat_batch)
            return deltaH.item()  # Return the scalar value of deltaH
# --------------------------
# Training with EMA
# --------------------------
def train_epoch_ema(model, ema_model, loader, optimizer, criterion, device, ema_decay=0.995):
    model.train(); total_loss=0; all_preds=[]; all_targets=[]
    for sat_batch, unsat_batch, y in loader:
        sat_batch, unsat_batch, y = sat_batch.to(device), unsat_batch.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(sat_batch, unsat_batch)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss+=loss.item()*y.size(0)
        all_preds.extend(out.detach().cpu().numpy().tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())
        if ema_model:
            with torch.no_grad():
                for ema_p, m_p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(m_p.data*(1-ema_decay))
    return total_loss/len(loader.dataset), np.array(all_preds), np.array(all_targets)

def evaluate_ema(model, loader, criterion, device):
    model.eval(); # Force dropout layers to stay active
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # keep dropout active during evaluation

    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for sat_batch, unsat_batch, y in loader:
            sat_batch, unsat_batch, y = sat_batch.to(device), unsat_batch.to(device), y.to(device)
            out = model(sat_batch, unsat_batch)
            total_loss+=criterion(out,y).item()*y.size(0)
            all_preds.extend(out.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
    return total_loss/len(loader.dataset), np.array(all_preds), np.array(all_targets)

# --------------------------
# Cross-Validation
# --------------------------
def cross_validation_train(df, n_splits=5):
    dataset = prepare_dataset(df)
    all_targets = np.array([y for (_,_,y) in dataset]).reshape(-1,1)
    scaler = StandardScaler(); scaler.fit(all_targets)
    dataset_scaled = [(g1,g2,float(scaler.transform([[y]])[0,0])) for g1,g2,y in dataset]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_preds, fold_targets = [], []
    fold_train_r2_list, fold_train_rmse_list = [], []
    fold_val_r2_list, fold_val_rmse_list = [], []
    fold_train_loss_list, fold_val_loss_list = [], []

    for fold,(train_idx,val_idx) in enumerate(kf.split(dataset_scaled),1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        train_data = [dataset_scaled[i] for i in train_idx]
        val_data = [dataset_scaled[i] for i in val_idx]

        train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_data, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)

        sample_in = train_data[0][0].x.size(1)
        model = SiameseGIN(sample_in,256,256,256,dropout=CONFIG['dropout']).to(CONFIG['device'])
        ema_model = copy.deepcopy(model).to(CONFIG['device'])
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        criterion = nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.7, min_lr=1e-6)

        best_val_r2 = -float('inf'); patience_counter=0; early_stop_patience=10

        train_loss_history, val_loss_history = [], []

        for epoch in range(1, CONFIG['num_epochs']+1):
            if epoch<=CONFIG['warmup_epochs']:
                lr=CONFIG['learning_rate']*epoch/CONFIG['warmup_epochs']
                for g in optimizer.param_groups: g['lr']=lr

            train_loss,_ ,_ = train_epoch_ema(model, ema_model, train_loader, optimizer, criterion, CONFIG['device'])
            val_loss, val_preds, val_targets = evaluate_ema(ema_model, val_loader, criterion, CONFIG['device'])

            # Compute Train metrics using EMA model
            _, train_preds, train_targets = evaluate_ema(ema_model, train_loader, criterion, CONFIG['device'])
            train_r2 = r2_score(scaler.inverse_transform(train_targets.reshape(-1,1)),
                                scaler.inverse_transform(train_preds.reshape(-1,1)))
            train_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(train_targets.reshape(-1,1)),
                                                    scaler.inverse_transform(train_preds.reshape(-1,1))))

            val_r2 = r2_score(scaler.inverse_transform(val_targets.reshape(-1,1)),
                              scaler.inverse_transform(val_preds.reshape(-1,1)))
            val_rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(val_targets.reshape(-1,1)),
                                                 scaler.inverse_transform(val_preds.reshape(-1,1))))

            scheduler.step(val_r2)
            print(f"Epoch {epoch:2d} | Train Loss {train_loss:.4f}, RMSE {train_rmse:.2f}, R² {train_r2:.4f} | "
                  f"Val Loss {val_loss:.4f}, RMSE {val_rmse:.2f}, R² {val_r2:.4f}")

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            if val_r2>best_val_r2: best_val_r2=val_r2; patience_counter=0; best_model_wts=copy.deepcopy(ema_model.state_dict())
            else: patience_counter+=1
            if patience_counter>=early_stop_patience: print(f"Early stopping at epoch {epoch}"); break

        ema_model.load_state_dict(best_model_wts)
        torch.save(best_model_wts,f"{CONFIG['save_dir']}/fold{fold}_best.pth")
        fold_preds.append(scaler.inverse_transform(val_preds.reshape(-1,1)).flatten())
        fold_targets.append(scaler.inverse_transform(val_targets.reshape(-1,1)).flatten())
        fold_train_r2_list.append(train_r2)
        fold_train_rmse_list.append(train_rmse)
        fold_val_r2_list.append(val_r2)
        fold_val_rmse_list.append(val_rmse)
        fold_train_loss_list.append(train_loss_history)
        fold_val_loss_list.append(val_loss_history)

    all_preds_val = np.concatenate(fold_preds)
    all_targets_val = np.concatenate(fold_targets)
    total_r2 = r2_score(all_targets_val, all_preds_val)
    total_rmse = np.sqrt(mean_squared_error(all_targets_val, all_preds_val))
    print(f"\n=== CV Completed: Average Test R²={total_r2:.4f}, RMSE={total_rmse:.3f} kJ/mol ===")

    # -------------------
    # Scatter plot (Val predictions vs targets, train metrics in text)
    # -------------------
    plt.figure(figsize=(6,6))
    plt.scatter(all_targets_val, all_preds_val, alpha=0.5, s=10)
    mn, mx = all_targets_val.min(), all_targets_val.max()
    plt.plot([mn,mx],[mn,mx],'--',color='red')
    plt.xlabel("True δH (kJ/mol)")
    plt.ylabel("Predicted δH (kJ/mol)")
    plt.title(f"CV Predictions — Train R²={np.mean(fold_train_r2_list):.4f}, | "
              f"Val R²={np.mean(fold_val_r2_list):.4f},")
    plt.grid(True); plt.tight_layout(); plt.show()

    # -------------------
    # Loss vs Epoch (average over folds)
    # -------------------
    avg_train_loss = np.mean(np.array(fold_train_loss_list), axis=0)
    avg_val_loss = np.mean(np.array(fold_val_loss_list), axis=0)
    plt.figure(figsize=(6,4))
    plt.plot(range(1,len(avg_train_loss)+1), avg_train_loss, label="Train Loss", marker='o')
    plt.plot(range(1,len(avg_val_loss)+1), avg_val_loss, label="Val Loss", marker='s')
    plt.xlabel("Epoch", fontsize=14); plt.ylabel("Loss", fontsize=14)
    plt.title("Average Train & Validation Loss vs Epoch (All Folds)")
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.show()

    return all_preds_val, all_targets_val

# --------------------------
# Main
# --------------------------
if __name__=="__main__":
    df = pd.read_csv(CONFIG['data_path'])
    preds, targets = cross_validation_train(df)
    # 推理解注释示例：
    # sat_smiles = ["C1CCCCC1"]
    # unsat_smiles = ["c1ccccc1"]
    # infer_preds = predict_delta_H_gnn2(sat_smiles, unsat_smiles, fig_path="inference_scatter.png")
    # print(infer_preds)
