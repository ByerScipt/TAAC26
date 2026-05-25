import os
import random
import copy
import logging
import time
from datetime import timedelta
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogFormatter:
    """统一训练日志格式。

    每条日志都会同时显示两种时间：当前机器时间和本次运行已耗时。训练任务常常跑
    几小时甚至更久，这样可以直接定位某个事件发生在几点，以及它距离启动过了多久。

    多行日志会把续行缩进到正文位置，堆栈、配置和多行统计信息读起来更整齐。
    """

    def __init__(self) -> None:
        # 记录计时起点，用来计算日志前缀里的已耗时。
        # 运行中可以通过 ``create_logger(...).reset_time()`` 重置。
        self.start_time: float = time.time()

    def format(self, record: logging.LogRecord) -> str:
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s" % (
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        # 多行日志的续行按正文位置缩进，避免和时间戳前缀混在一起。
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message)


def create_logger(filepath: str) -> logging.Logger:
    """创建训练和推理共用的 root logger。

    这里同时配置文件日志和控制台日志：

    * 文件日志写到 ``filepath``，记录 ``DEBUG`` 及以上级别，用于事后复盘。
    * 控制台日志只显示 ``INFO`` 及以上级别，避免训练时刷屏。

    函数会清空 root logger 上已经挂载的 handler。多次启动训练、notebook 反复
    调用或测试脚本重复初始化时，这一步能避免同一条日志被打印多遍。

    ``reset_time()`` 会挂到返回的 logger 上。数据加载和 schema 构建结束后，可以
    重新开始计时，把日志里的耗时集中反映训练阶段。

    参数：
        filepath: 日志文件路径，以 ``"w"`` 模式打开。

    返回：
        配置完成的 root ``logging.Logger``。
    """
    log_formatter = LogFormatter()

    file_handler = logging.FileHandler(filepath, "w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 允许调用方重置日志前缀中展示的已耗时计时器。
    def reset_time() -> None:
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time  # type: ignore[attr-defined]

    return logger


class EarlyStopping:
    """根据验证集指标控制早停和最优模型保存。

    当前比赛主指标是 AUC，所以这里按“指标越高越好”处理。新的 ``score`` 只有超过
    ``best_score + delta`` 才算一次有效提升；连续 ``patience`` 次没有提升后，
    ``early_stop`` 会被置为 ``True``。

    每次有效提升都会做两件事：把当前 ``state_dict`` 深拷贝到内存，方便进程内
    继续使用；同时写到 ``checkpoint_path``，供后续推理和线上提交使用。
    ``best_saved_score`` 记录最近一次写盘对应的分数，上层保存逻辑会用它减少重复
    checkpoint 写入。

    属性：
        checkpoint_path: 最优模型参数的保存路径。
        patience: 连续多少次验证无提升后停止。
        verbose: 写 checkpoint 时是否额外输出日志。
        counter: 当前连续无提升次数。
        best_score: 当前最佳指标。
        early_stop: 外层训练循环读取这个标记来决定是否退出。
        delta: 判定为提升所需的最小幅度。
        best_model: 内存中的最优模型参数快照。
        best_saved_score: 最近一次写盘的最佳指标。
        best_extra_metrics: 最优点附带的其他指标，例如 logloss。
        label: 日志前缀，用来区分不同 early stopping 实例。
    """

    def __init__(
        self,
        checkpoint_path: str,
        label: str = "",
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0,
    ) -> None:
        self.checkpoint_path: str = checkpoint_path
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.delta: float = delta
        self.best_model: Optional[Dict[str, torch.Tensor]] = None
        self.best_saved_score: float = 0.0
        self.best_extra_metrics: Optional[Dict[str, Any]] = None
        self.label: str = label
        if self.label != "":
            self.label += " "

    def _is_not_improved(self, score: float) -> bool:
        """判断当前分数是否没有带来有效提升。

        调用前 ``best_score`` 已经被首次验证分数初始化。返回 ``True`` 时，上层会
        累加 patience 计数。
        """
        assert self.best_score is not None, "call __call__ first to seed best_score"
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(
        self,
        score: float,
        model: nn.Module,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """处理一次新的验证结果。

        分支逻辑如下：

        1. 第一次验证：初始化最佳分数，并立即保存一份 checkpoint。
        2. 分数无提升：累加 ``counter``，达到 ``patience`` 后触发早停。
        3. 分数提升：重置 ``counter``，更新最佳分数、辅助指标和模型快照。

        参数：
            score: 本次验证的主指标，例如 AUC。
            model: 需要保存参数快照的模型。
            extra_metrics: 和主指标同一步产生的辅助指标，会原样记录。
        """
        if self.best_score is None:
            self.best_score = score
            self.best_extra_metrics = extra_metrics
            self.best_saved_score = 0.0
            self.save_checkpoint(score, model)
            self.best_model = copy.deepcopy(model.state_dict())
        elif self._is_not_improved(score):
            self.counter += 1
            logging.info(f'{self.label}earlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            logging.info(f'{self.label}earlyStopping counter reset!')
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_extra_metrics = extra_metrics
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module) -> None:
        """保存当前模型参数。

        写盘前会创建父目录。这里保存的是 ``state_dict``，不包含优化器和 scheduler。
        保存成功后会更新 ``best_saved_score``，方便外层逻辑判断这次调用是否真的
        产生了新 checkpoint。

        参数：
            score: 当前参数对应的验证分数。
            model: 需要保存参数的模型。
        """
        if self.verbose:
            logging.info('Validation score increased. Saving model ...')
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_saved_score = score


def set_seed(seed: int) -> None:
    """设置训练中常见随机源的种子。

    覆盖 Python ``random``、``PYTHONHASHSEED``、NumPy、PyTorch CPU 随机数和
    CUDA 随机数，并打开 cuDNN deterministic 模式。

    GPU 训练很难做到完全逐 bit 复现，尤其是不同驱动、不同 CUDA kernel 和并行
    reduction 都会影响细节。这个函数用于降低随机性，方便比较实验趋势。

    参数：
        seed: 所有随机源共用的整数种子。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """计算二分类 Focal Loss。

    公式为 ``FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)``。它会降低
    容易样本的权重，让训练更关注当前模型分错或置信度低的样本。

    参数：
        logits: 形状 ``(N,)``，模型输出的原始 logit。
        targets: 形状 ``(N,)``，二分类标签，取值 ``0`` 或 ``1``。
        alpha: 正类权重。
        gamma: 聚焦参数；数值越大，容易样本的 loss 权重越低。
        reduction: ``'mean'``、``'sum'`` 或 ``'none'``。
    """
    p = torch.sigmoid(logits)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * bce_loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss
