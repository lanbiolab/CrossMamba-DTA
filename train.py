import torch
import torch.nn as nn
import logging
import os
import json
import numpy as np
from torch.utils.data import DataLoader
from dataloader import DTADataset, collate_fn
from model import DTA
# 引入新写的联合损失函数
from metrics import concordance_index, get_rm2

torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format='%(message)s',
                    handlers=[logging.FileHandler("training_mamba_sota_512_8_60_epoch.txt", mode='w'), logging.StreamHandler()])
logger = logging.getLogger()

class EarlyStopping:
    def __init__(self, patience=15, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False

    def __call__(self, current_mse):
        if current_mse < self.best_score - self.delta:
            self.best_score = current_mse
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def evaluate(model, data_loader, criterion, device, std_val, mean_val, calc_metrics=False, prefix="Val"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    y_t_all, y_p_all = [], []

    with torch.no_grad():
        for batch in data_loader:
            d_seq, p_seq, d_mask, p_mask, labels = batch

            d_seq, p_seq = d_seq.to(device, non_blocking=True), p_seq.to(device, non_blocking=True)
            d_mask, p_mask = d_mask.to(device, non_blocking=True), p_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                preds = model(d_seq, p_seq, d_mask, p_mask)
                loss = criterion(preds, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            y_p_all.extend(preds.float().cpu().numpy().flatten())
            y_t_all.extend(labels.float().cpu().numpy())

    avg_loss = total_loss / total_samples

    # 还原归一化算真实的指标
    y_t_orig = np.array(y_t_all) * std_val + mean_val
    y_p_orig = np.array(y_p_all) * std_val + mean_val
    mse = np.mean((y_t_orig - y_p_orig) ** 2)

    ci, rm2 = None, None
    if calc_metrics:
        ci = concordance_index(y_t_orig, y_p_orig)
        rm2 = get_rm2(y_t_orig, y_p_orig)

    return avg_loss, mse, ci, rm2


def train_fold(config, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{'=' * 20} Starting Fold {fold} {'=' * 20}")

    dataset_dir = os.path.join(config["data_base_path"], config["dataset_name"])
    stats_path = os.path.join(dataset_dir, f'fold_{fold}_stats.json')
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        mean_val, std_val = stats['mean'], stats['std']
    except:
        mean_val, std_val = 0.0, 1.0

    drugs_dir = os.path.join(dataset_dir, "drugs")
    proteins_dir = os.path.join(dataset_dir, "proteins")
    train_index_path = os.path.join(dataset_dir, f"train_fold_{fold}_index.pt")
    val_index_path = os.path.join(dataset_dir, f"val_fold_{fold}_index.pt")
    test_index_path = os.path.join(dataset_dir, f"test_fold_{fold}_index.pt")

    train_dataset = DTADataset(train_index_path, drugs_dir, proteins_dir)
    val_dataset = DTADataset(val_index_path, drugs_dir, proteins_dir)
    test_dataset = DTADataset(test_index_path, drugs_dir, proteins_dir)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              collate_fn=collate_fn, drop_last=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, pin_memory=True,
                            num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn, pin_memory=True,
                             num_workers=2)

    model = DTA(hidden_dim=config.get('hidden_dim', 512)).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-3)

    # 【方案 1：引入联合损失】训练用 MSE+Rank，评估只管 MSE
    # train_criterion = DTACombinedLoss(weight_mse=1.0, weight_rank=0.5).to(device)
    train_criterion = nn.MSELoss().to(device)
    eval_criterion = nn.MSELoss().to(device)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'],
                                                           eta_min=config.get('min_lr', 1e-6))

    early_stopping = EarlyStopping(patience=config['patience'])
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            d_seq, p_seq, d_mask, p_mask, labels = batch
            d_seq, p_seq = d_seq.to(device), p_seq.to(device)
            d_mask, p_mask = d_mask.to(device), p_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(d_seq, p_seq, d_mask, p_mask)
                loss = train_criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

        scheduler.step()

        val_loss, val_mse, val_ci, val_rm2 = evaluate(model, val_loader, eval_criterion, device, std_val, mean_val, calc_metrics=False, prefix="Val")

        # logger.info(
        #     f"Fold {fold} | Ep {epoch:03d} | "
        #     f"Train Loss (Combined): {np.mean(epoch_losses):.4f} | "
        #     f"Val MSE: {val_mse:.4f} | Val CI: {val_ci:.4f} | Val Rm2: {val_rm2:.4f}"
        # )
        logger.info(
            f"Fold {fold} | Ep {epoch:03d} | "
            f"Train Loss: {np.mean(epoch_losses):.4f} | "
            f"Val MSE: {val_mse:.4f}"
        )

        if early_stopping(val_mse):
            torch.save(model.state_dict(), f"kiba_best_model_fold_{fold}.pt")

        if early_stopping.early_stop:
            logger.info(f"==> 早停机制触发，停止于第 {epoch} 轮 (Best Val MSE: {early_stopping.best_score:.4f})")
            break

    model.load_state_dict(torch.load(f"kiba_best_model_fold_{fold}.pt"))
    test_loss, test_mse, test_ci, test_rm2 = evaluate(model, test_loader, eval_criterion, device, std_val, mean_val, calc_metrics=True, prefix="Test")

    logger.info(f"\n--- Fold {fold} 最终测试结果 ---")
    logger.info(f"Test MSE: {test_mse:.4f} | Test CI: {test_ci:.4f} | Test Rm2: {test_rm2:.4f}\n")

    return test_mse, test_ci, test_rm2


def main_cv(config):
    logger.info(f"Task: {config['dataset_name']} | Cross-Mamba SOTA Sprint")
    results_mse, results_ci, results_rm2 = [], [], []

    for fold in range(5):
        test_mse, test_ci, test_rm2 = train_fold(config, fold)
        results_mse.append(test_mse)
        results_ci.append(test_ci)
        results_rm2.append(test_rm2)

    logger.info(f"\n{'=' * 15} 5-Fold Cross Validation Final Results {'=' * 15}")
    logger.info(f"Dataset: {config['dataset_name'].upper()}")
    logger.info(f"Final Test MSE : {np.mean(results_mse):.4f} ± {np.std(results_mse):.4f}")
    logger.info(f"Final Test CI  : {np.mean(results_ci):.4f} ± {np.std(results_ci):.4f}")
    logger.info(f"Final Test Rm2 : {np.mean(results_rm2):.4f} ± {np.std(results_rm2):.4f}")
    logger.info(f"{'=' * 65}")


if __name__ == '__main__':
    config = {
        "dataset_name": "davis",
        "data_base_path": "./processed_data",
        "batch_size": 256,

        "epochs": 1000,
        "lr": 3e-5,
        "min_lr": 1e-6,
        "hidden_dim": 512,
        "patience": 60
    }
    main_cv(config)