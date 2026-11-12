# utils/metrics.py

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def concordance_index(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]
    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult
    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    return mult / float(y_obs_sq * y_pred_sq)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


class EvidentialRegressionLoss(nn.Module):
    def __init__(self, coeff_lambda=0.1):
        super(EvidentialRegressionLoss, self).__init__()
        self.coeff_lambda = coeff_lambda

    def NIG_NLL(self, y, gamma, v, alpha, beta):
        two_blambda = 2 * beta * (1 + v)
        nll = 0.5 * torch.log(np.pi / v) \
              - alpha * torch.log(two_blambda) \
              + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_blambda) \
              + torch.lgamma(alpha) \
              - torch.lgamma(alpha + 0.5)
        return nll

    def NIG_Reg(self, y, gamma, v, alpha):
        error = torch.abs(y - gamma)
        evidence = 2 * v + alpha
        return error * evidence

    def forward(self, y, gamma, v, alpha, beta):
        loss_nll = self.NIG_NLL(y, gamma, v, alpha, beta)
        loss_reg = self.NIG_Reg(y, gamma, v, alpha)

        return torch.mean(loss_nll + self.coeff_lambda * loss_reg)


class VarianceMatchingLoss(nn.Module):
    """
    方差匹配 MSE 损失函数 (改进版)
    针对标准化标签，强制模型预测值的分布方差向 1.0 靠拢。
    解决小 Batch 内部真实方差抖动导致的训练不稳定问题。
    """
    def __init__(self, lamda=0.1):
        super(VarianceMatchingLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lamda = lamda

    def forward(self, preds, targets):
        # 1. 基础 MSE 损失
        loss_mse = self.mse(preds, targets)

        # 2. 锚定全局方差 (目标方差恒定为 1.0)
        if preds.size(0) > 1:
            var_pred = torch.var(preds, unbiased=False)
            loss_var = torch.abs(var_pred - 1.0)
        else:
            loss_var = torch.tensor(0.0, device=preds.device)

        # 3. 组合损失
        total_loss = loss_mse + self.lamda * loss_var
        return total_loss
