import math
import inspect
from dataclasses import dataclass, field, asdict
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits, targets: 0/1
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # p_t
        
        # Focal Loss 公式: -alpha * (1-pt)^gamma * log(pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Causal Mask (只看过去)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Causal mask: ensure distinctness and causality
        att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att

class FFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FFN(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask)
        x = x + y
        x = x + self.ffn(self.ln_2(x))
        return x, att

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        hidden_dim = input_dim * 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class FollowUpEncoder(nn.Module):
    """
    处理一次随访中的混合数据：
    1. 数值特征 (Numerical): 通过 Linear 层投影
    2. 类别特征 (Categorical): 通过 Embedding 层查

    Args:
        x_num (torch.Tensor): 数值特征输入张量。
            Shape: (batch_size, seq_len, num_numerical_features)
            Type: torch.float32
        x_cat (torch.Tensor): 类别特征输入张量(对应词表中的 ID)。
            Shape: (batch_size, seq_len, num_categorical_features)
            Type: torch.long

    Returns:
        torch.Tensor: 融合并映射到高维空间的随访特征向量。
            Shape: (batch_size, seq_len, n_embd)

    Example:
        batch_size=2, seq_len=3 (3天), num_numerical_features=2 (e.g. ALT, WBC)
        num_categorical_features=2 (e.g. 乙肝阴阳性, HPV阴阳性)
        x_num = torch.tensor([
                [[45.5, 12.0], [120.0, 30.5], [40.0, 10.0]], # 病人1的三天记录
                [[50.0, 15.0], [55.0, 18.0],  [60.0, 20.0]]  # 病人2的三天记录
            ])

        词表: 0=阴性, 1=阳性, 2=缺失
        x_cat = torch.tensor([
                [[0, 1], [0, 0], [1, 1]], # 病人1: 阴/阳 -> 阴/阴 -> 阳/阳
                [[0, 0], [2, 0], [0, 0]]  # 病人2: 阴/阴 -> 缺/阴 -> 阴/阴
             ])

    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        
        self.cat_embeddings = nn.ModuleList()
        cat_output_dim = 0

        # 每个类别特征分配 16 维 (预期：16 * 7 = 112 维)
        emb_dim_per_feature = 16

        if config.categorical_cardinalities:
            for num_classes in config.categorical_cardinalities:
                self.cat_embeddings.append(nn.Embedding(num_classes, emb_dim_per_feature))
                cat_output_dim += emb_dim_per_feature

        # 剩余空间给数值特征
        num_output_dim = config.n_embd - cat_output_dim

        if num_output_dim <= 0:
             raise ValueError(f"n_embd 设置过小")
        
        if config.num_numerical_features > 0:
            self.num_mlp = MLP(
                input_dim=config.num_numerical_features,
                output_dim=num_output_dim,
                dropout=config.dropout
            )
        
        self.final_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x_num, x_cat):
        """
        x_num: [Batch, Time, Num_Features] (Float)
        x_cat: [Batch, Time, Cat_Features] (Long/Int)
        """

        if self.num_mlp is not None:
            ref_param = next(self.num_mlp.parameters())
            x_num = x_num.to(device=ref_param.device, dtype=ref_param.dtype)

        if len(self.cat_embeddings) > 0:
            ref_emb = self.cat_embeddings[0].weight
            x_cat = x_cat.to(device=ref_emb.device)

        features = []
        
        # 处理数值
        if self.num_mlp is not None:
            x1 = self.num_mlp(x_num) # -> (B, T, n_embd/2)
            features.append(x1)
        
        # 处理类别
        if len(self.cat_embeddings) > 0:
            cat_embs = []
            for i, emb_layer in enumerate(self.cat_embeddings):
                # 取出第 i 个指标的数据进行查表
                cat_embs.append(emb_layer(x_cat[..., i])) 
            
            x2 = torch.cat(cat_embs, dim=-1) # 拼接所有类别向量
            features.append(x2)
            
        # 融合 -> [Batch, Time, n_embd]
        x = torch.cat(features, dim=-1) 
        x = self.final_proj(x)
        x = self.dropout(x)
        return x

class DayEncoding(nn.Module):
    """
    对 "天数" 进行位置编码
    """
    def __init__(self, config, max_timescale = 10000.0):
        super().__init__()
        self.n_embd = config.n_embd
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(max_timescale) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        """
        x: Tensor, shape [batch, time, 1]
        """
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x * self.div_term) 
        y[..., 1::2] = torch.cos(x * self.div_term) 
        
        y = self.linear(y)
        return y


@dataclass
class Config:
    n_layer: int = 12
    n_head: int = 12       
    n_embd: int = 768
    block_size: int = 512   # 序列长度 (随访次数上限)
    num_numerical_features: int = 40

    # 类别指标的类别数列表 (例如: 3个指标，每个都有 阴/阳/缺 3种状态)
    categorical_cardinalities: list = field(default_factory=lambda: [3, 3, 3])           

    num_targets: int = 34 # 需要预测的二分类指标数量

    # 预测窗口配置
    # 例如: [4, 10, -1] 代表 3 个窗口 (1-4天, 5-14天, >14天)
    prediction_windows: list = field(default_factory=lambda: [4, 10, -1])

    dropout: float = 0.0
    bias: bool = True 

    hierarchical_n_layer: int = 2 # 生成式解码器的层数

    def to_dict(self):
        return asdict(self)
    
class TaskHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_targets = config.num_targets
        self.num_windows = len(config.prediction_windows)
        total_output_dim = self.num_targets * self.num_windows
        self.proj = nn.Linear(config.n_embd, total_output_dim)

    def forward(self, x):
        """
        Args:
            x: Transformer 输出特征 [Batch, Time, n_embd]
        Returns:
            logits: [Batch, Time, Num_Targets, Num_Windows]
        """
        B, T, C = x.size()
        
        # -> [Batch, Time, Num_Targets * Num_Windows]
        flat_logits = self.proj(x)
        
        # -> [Batch, Time, Num_Targets, Num_Windows]
        logits = flat_logits.view(B, T, self.num_targets, self.num_windows)
        
        return logits
    
class PredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            followup_encoder = FollowUpEncoder(config),
            pos_embd = DayEncoding(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.num_windows = len(config.prediction_windows)

        self.cls_head = TaskHead(config)
        self.loss_fct = BinaryFocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        # init weights
        self.apply(self._init_weights)
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, num_x, cat_x, days, labels=None, padding_mask=None):
        """
        Args:
            num_x:   [Batch, Time, Num_Numerical_Features] (Float)
            cat_x:   [Batch, Time, Num_Categorical_Features] (Long)
            days:    [Batch, Time] (Float, 单位: 天) 使用float而不是int是方便位置编码
            labels: [Batch, Time, Num_Targets, Num_Windows] (Int, 0/1标签)
            padding_mask: [Batch, Time] (Bool/Int). 1=有效数据, 0=Padding。

        Returns:
            logits:         [Batch, Time, Num_Targets, Num_Windows]
            loss:           Scalar
            all_attentions: List[Tensor], 包含每一层的注意力权重
        """
        device = num_x.device
        b, t, _ = num_x.size()

        followup_emb = self.transformer.followup_encoder(num_x, cat_x)
        # days: [Batch, Time] -> [Batch, Time, 1]
        day_emb = self.transformer.pos_embd(days.unsqueeze(-1))
        
        x = followup_emb + day_emb
        x = self.transformer.drop(x)
        
        # 默认全为 1 (有效)
        if padding_mask is None:
            padding_mask = torch.ones((b, t), device=device, dtype=torch.long)
        is_valid = (padding_mask > 0) # [B, T]

        # Padding Mask: [B, 1, 1, T]
        mask_padding = is_valid.view(b, 1, 1, t).float()
        # Causal Mask: [1, 1, T, T]
        mask_causal = torch.tril(torch.ones(t, t, device=device)).view(1, 1, t, t)

        attn_mask = mask_padding * mask_causal

        diag_mask = torch.eye(t, device=device).view(1, 1, t, t)
        attn_mask = torch.max(attn_mask, diag_mask)

        all_attentions = []

        for block in self.transformer.h:
            x, att = block(x, attn_mask=attn_mask)
            all_attentions.append(att)
        x = self.transformer.ln_f(x)      
        all_attentions = torch.stack(all_attentions)

        logits = None
        loss = None
        logits = self.cls_head(x)


        if labels is not None:
            # loss_fct = nn.BCEWithLogitsLoss(reduction='none') 
            # loss = loss_fct(logits, labels.float())

            loss = self.loss_fct(logits, labels.float())
            
            
            # 时间步掩码 (Padding Mask)
            # [Batch, Time, 1, 1]
            mask_time = is_valid.view(b, t, 1, 1).float()

            # 标签有效性掩码 (Label Valid Mask)
            # 过滤掉 dataset.py 中填入的 -1.0
            # [Batch, Time, Num_Targets, Num_Windows]
            mask_label = (labels != -1).float()
            
            # 组合掩码
            final_mask = mask_time * mask_label

            loss = loss * final_mask

            # 计算平均 Loss
            num_valid_elements = final_mask.sum()
            loss = loss.sum() / (num_valid_elements + 1e-6)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            attentions=tuple(all_attentions),
            hidden_states=None 
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # standard configuration
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def predict(self, num_x, cat_x, days):
        """
        推理模式: 返回概率 (0~1) 和 二值化预测 (0/1)
        """
        self.eval()
        logits, _, _ = self(num_x, cat_x, days)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        return preds, probs



class HierarchicalHeadBlock(nn.Module):
    """
    专门用于处理层级关系的微型 Transformer Block。
    它将被应用在每一个时间步上。
    序列长度 = 1 (History Context) + Num_Windows
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn = FFN(config)

    def forward(self, x, attn_mask):
        # x shape: [Batch * Time, Num_Hierarchy_Nodes, n_embd]
        y, _ = self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + y
        x = x + self.ffn(self.ln_2(x))
        return x

class HierarchicalDensePredictionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Temporal Transformer (主骨干): 处理历史信息的时间序列
        self.transformer = nn.ModuleDict(dict(
            followup_encoder = FollowUpEncoder(config),
            pos_embd = DayEncoding(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        self.num_windows = len(config.prediction_windows)
        self.num_targets = config.num_targets

        # 2. 层级相关组件
        # Window Queries: 代表 [Window 1, Window 2, Window 3] 的语义向量
        self.window_queries = nn.Parameter(torch.randn(self.num_windows, config.n_embd))
        
        # 3. Hierarchical Head (层级头)
        self.hierarchical_blocks = nn.ModuleList(
            [HierarchicalHeadBlock(config) for _ in range(config.hierarchical_n_layer)]
        )
        self.ln_hierarchical = LayerNorm(config.n_embd, bias=config.bias)

        # 4. 最终分类头
        # 共享权重，用于从每个 Window Query 的输出中预测指标
        self.cls_head = nn.Linear(config.n_embd, self.num_targets)
        
        self.loss_fct = BinaryFocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """
        获取模型参数总数
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, num_x, cat_x, days, labels=None, padding_mask=None):
        """
        Args:
            num_x: [Batch, Time, Num_Features]
            labels: [Batch, Time, Num_Targets, Num_Windows]
        """
        device = num_x.device
        b, t, _ = num_x.size()

        # -----------------------------------------------------------
        # Step 1: Temporal Encoding (处理历史，获得每一天的上下文)
        # -----------------------------------------------------------
        history_emb = self.transformer.followup_encoder(num_x, cat_x)
        day_emb = self.transformer.pos_embd(days.unsqueeze(-1))
        
        x_history = history_emb + day_emb
        x_history = self.transformer.drop(x_history)

        # 构建主骨干的时间掩码 (Standard Causal Mask for Time)
        if padding_mask is None:
            padding_mask = torch.ones((b, t), device=device, dtype=torch.long)
            
        mask_padding = padding_mask.view(b, 1, 1, t).float()
        mask_temporal_causal = torch.tril(torch.ones(t, t, device=device)).view(1, 1, t, t)
        attn_mask_temporal = mask_padding * mask_temporal_causal
        attn_mask_temporal = torch.max(attn_mask_temporal, torch.eye(t, device=device).view(1, 1, t, t)) # safe guard

        # Run Temporal Transformer
        for block in self.transformer.h:
            x_history, _ = block(x_history, attn_mask=attn_mask_temporal)
        x_history = self.transformer.ln_f(x_history)
        
        # 此时 x_history: [Batch, Time, n_embd]
        # 它包含了每一个时间步对过去历史的总结。要在每一个时间步上展开层级预测。

        # -----------------------------------------------------------
        # Step 2: Prepare for Hierarchical Generation (每个时间步展开)
        # -----------------------------------------------------------
        # 我们将 Batch 和 Time 维度合并，视为独立的样本进行层级处理
        # [Batch * Time, n_embd]
        flat_history_ctx = x_history.reshape(b * t, self.config.n_embd)
        
        # 准备 Window Queries
        # [Num_Windows, n_embd] -> [Batch * Time, Num_Windows, n_embd]
        w_queries = self.window_queries.unsqueeze(0).expand(b * t, -1, -1)
        
        # 构造层级序列: [History_Context, Window_1, Window_2, Window_3]
        # Length = 1 + Num_Windows
        # [Batch * Time, 1, n_embd]
        flat_history_ctx_unsqueezed = flat_history_ctx.unsqueeze(1)
        
        # [Batch * Time, 1 + Num_Windows, n_embd]
        h_input = torch.cat([flat_history_ctx_unsqueezed, w_queries], dim=1)
        
        seq_len_h = h_input.size(1) # 1 + Num_Windows

        # -----------------------------------------------------------
        # Step 3: PAAM Mask Construction (在层级维度上)
        # -----------------------------------------------------------
        # 这里的 Mask 决定了 Window 之间如何可见。
        # 逻辑：
        # - History Context (Pos 0): 只能看自己
        # - Window 1 (Pos 1): 能看 History (Pos 0) + 自己
        # - Window 2 (Pos 2): 能看 History (Pos 0) + W1 (Pos 1) + 自己
        # ...
        
        # [1, 1, Seq_Len_H, Seq_Len_H]
        paam_mask = torch.tril(torch.ones(seq_len_h, seq_len_h, device=device)).view(1, 1, seq_len_h, seq_len_h)
        
        # -----------------------------------------------------------
        # Step 4: Hierarchical Head Forward
        # -----------------------------------------------------------
        h_x = h_input
        for block in self.hierarchical_blocks:
            h_x = block(h_x, attn_mask=paam_mask)
        h_x = self.ln_hierarchical(h_x)
        
        # -----------------------------------------------------------
        # Step 5: Prediction & Reshape
        # -----------------------------------------------------------
        # 只取后面 Window 部分的输出，忽略第一个 History Token 的输出
        # [Batch * Time, Num_Windows, n_embd]
        window_outputs = h_x[:, 1:, :] 
        
        # [Batch * Time, Num_Windows, Num_Targets]
        logits_flat = self.cls_head(window_outputs)
        
        # 恢复维度: [Batch, Time, Num_Windows, Num_Targets]
        logits = logits_flat.view(b, t, self.num_windows, self.num_targets)
        
        # 调整为与 Label 一致: [Batch, Time, Num_Targets, Num_Windows]
        logits = logits.permute(0, 1, 3, 2)

        # -----------------------------------------------------------
        # Step 6: Loss Calculation 
        # -----------------------------------------------------------
        loss = None
        if labels is not None:
            # labels: [Batch, Time, Num_Targets, Num_Windows]
            
            loss = self.loss_fct(logits, labels.float())
            
            # Mask 1: 无效的标签 (-1)
            mask_label_valid = (labels != -1).float()
            
            # Mask 2: Padding 的时间步 (padding_mask)
            # padding_mask: [Batch, Time] -> [Batch, Time, 1, 1]
            mask_time_valid = padding_mask.view(b, t, 1, 1).float()
            
            final_mask = mask_label_valid * mask_time_valid
            
            loss = loss * final_mask
            loss = loss.sum() / (final_mask.sum() + 1e-6)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        if hasattr(self, 'window_queries'):
             decay.add('window_queries')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def predict(self, num_x, cat_x, days, padding_mask=None):
        self.eval()
        outputs = self(num_x, cat_x, days, padding_mask=padding_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        return preds, probs
    


class ExplicitHierarchicalModel(nn.Module):
    """
    显式自回归模型。
    对应原论文 PAAM-HiA-T5 的 "Generative" 模式。
    
    机制：
    - Input: [History Context, SOS_Token, Label_Emb_W1, Label_Emb_W2]
    - Output: [Prediction_W1, Prediction_W2, Prediction_W3]
    
    特点：
    - 训练时使用 Teacher Forcing (输入真实标签)。
    - 推理时使用 Auto-regressive Loop (输入上一步的预测)。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_windows = len(config.prediction_windows)
        self.num_targets = config.num_targets

        # 1. 历史编码器 (复用)
        self.transformer = nn.ModuleDict(dict(
            followup_encoder = FollowUpEncoder(config),
            pos_embd = DayEncoding(config),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        # 2. 显式生成的组件
        # [SOS] Token: 序列开始的标志
        self.start_token = nn.Parameter(torch.randn(1, 1, config.n_embd))
        
        # 标签投影层: 将 Num_Targets 维度的 0/1 向量映射为 n_embd 维度的输入向量
        self.label_encoder = nn.Linear(self.num_targets, config.n_embd)
        
        # 层级解码器 (Hierarchical Decoder)
        # 注意: 这里使用 Causal Mask 来防止作弊 (只能看过去)
        self.hierarchical_blocks = nn.ModuleList(
            [HierarchicalHeadBlock(config) for _ in range(config.hierarchical_n_layer)]
        )
        self.ln_hierarchical = LayerNorm(config.n_embd, bias=config.bias)

        # 3. 输出头
        self.cls_head = nn.Linear(config.n_embd, self.num_targets)
        
        # 4. Loss
        self.loss_fct = BinaryFocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, num_x, cat_x, days, labels=None, padding_mask=None):
        """
        根据 labels 是否为 None，自动切换 Training (Teacher Forcing) 或 Inference (Auto-regressive) 模式
        """
        device = num_x.device
        b, t, _ = num_x.size()

        # --- Step 1: 编码历史 (与之前相同) ---
        history_emb = self.transformer.followup_encoder(num_x, cat_x)
        day_emb = self.transformer.pos_embd(days.unsqueeze(-1))
        x_history = history_emb + day_emb
        x_history = self.transformer.drop(x_history)

        if padding_mask is None:
            padding_mask = torch.ones((b, t), device=device, dtype=torch.long)
            
        mask_padding = padding_mask.view(b, 1, 1, t).float()
        mask_temporal = torch.tril(torch.ones(t, t, device=device)).view(1, 1, t, t)
        attn_mask_temporal = mask_padding * mask_temporal
        attn_mask_temporal = torch.max(attn_mask_temporal, torch.eye(t, device=device).view(1, 1, t, t))

        for block in self.transformer.h:
            x_history, _ = block(x_history, attn_mask=attn_mask_temporal)
        x_history = self.transformer.ln_f(x_history)
        
        # [Batch * Time, 1, n_embd]
        flat_history_ctx = x_history.reshape(b * t, self.config.n_embd).unsqueeze(1)

        # =====================================================================
        # 分支逻辑：训练 vs 推理
        # =====================================================================
        
        if labels is not None:
            # -----------------------------------------------------------
            # Mode A: 训练模式 (Teacher Forcing)
            # -----------------------------------------------------------
            # 我们直接使用 Ground Truth 作为输入
            # Labels Shape: [Batch, Time, Targets, Windows]
            
            # 1. 准备 Labels 输入
            # Flatten -> [Batch * Time, Targets, Windows]
            flat_labels = labels.reshape(b * t, self.num_targets, self.num_windows)
            # Transpose -> [Batch * Time, Windows, Targets]
            target_seq = flat_labels.permute(0, 2, 1).float()
            
            # 处理无效标签 (-1): 在Embedding前将其置为0，避免NaN (Loss计算时会再次Mask掉)
            safe_target_seq = target_seq.clone()
            safe_target_seq[safe_target_seq == -1] = 0.0
            
            # 2. 构造 Decoder 输入序列
            # Input Sequence: [SOS, GT_W1, GT_W2] -> 预测 [Pred_W1, Pred_W2, Pred_W3]
            # 我们只需要前 (Num_Windows - 1) 个 GT 作为输入
            # [Batch*Time, Num_Windows-1, n_embd]
            if self.num_windows > 1:
                input_label_embs = self.label_encoder(safe_target_seq[:, :-1, :])
                
                # 拼接 SOS Token
                # sos: [Batch*Time, 1, n_embd]
                sos = self.start_token.expand(b * t, -1, -1)
                
                # Decoder Input: [Batch*Time, Num_Windows, n_embd]
                # 序列: [SOS, Emb(W1), Emb(W2)]
                decoder_input = torch.cat([sos, input_label_embs], dim=1)
            else:
                # 只有一个窗口，输入只有 SOS
                decoder_input = self.start_token.expand(b * t, -1, -1)

            # 3. 拼接 History Context
            # Full Input: [Ctx, SOS, Emb(W1), Emb(W2)]
            h_input = torch.cat([flat_history_ctx, decoder_input], dim=1)
            
            # 4. 构建 Mask (PAAM / Causal)
            seq_len = h_input.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
            
            # 5. 前向传播
            h_x = h_input
            for block in self.hierarchical_blocks:
                h_x = block(h_x, attn_mask=causal_mask)
            h_x = self.ln_hierarchical(h_x)
            
            # 6. 预测与 Loss
            # 忽略 History Context (pos 0)，只取后面的输出
            # Output: [Pred_W1(from SOS), Pred_W2(from W1), Pred_W3(from W2)]
            window_outputs = h_x[:, 1:, :] 
            
            # [Batch*Time, Windows, Targets]
            logits_flat = self.cls_head(window_outputs)
            
            # 恢复维度 [B, T, Targets, Windows]
            logits = logits_flat.view(b, t, self.num_windows, self.num_targets).permute(0, 1, 3, 2)
            
            # 计算 Loss (与之前一致)
            loss = self.loss_fct(logits, labels.float())
            mask_label_valid = (labels != -1).float()
            mask_time_valid = padding_mask.view(b, t, 1, 1).float()
            final_mask = mask_label_valid * mask_time_valid
            loss = loss * final_mask
            loss = loss.sum() / (final_mask.sum() + 1e-6)
            
            # 为了保持接口一致，这里返回 [B, 1, Targets, Windows] 格式的 logits (针对 metrics)
            # 在 Evaluate 阶段，我们通常只关心最后的预测
            # 为了在 predict() 中能拿到正确结果，这里返回完整的 logits
            # 外部 metrics 逻辑会处理 shape
            return SequenceClassifierOutput(loss=loss, logits=logits)

        else:
            # -----------------------------------------------------------
            # Mode B: 推理模式 (Auto-regressive Loop)
            # -----------------------------------------------------------
            # 没有 Labels，必须一步步生成
            
            # 1. 初始输入: [History, SOS]
            sos = self.start_token.expand(b * t, -1, -1)
            # current_input: [Batch*Time, 2, n_embd]
            current_input = torch.cat([flat_history_ctx, sos], dim=1)
            
            outputs_list = []
            
            # 2. 自回归循环
            for i in range(self.num_windows):
                seq_len = current_input.size(1)
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)
                
                # Pass through blocks
                h_x = current_input
                for block in self.hierarchical_blocks:
                    h_x = block(h_x, attn_mask=causal_mask)
                h_x = self.ln_hierarchical(h_x)
                
                # 取最后一个 Token 的输出 (当前步的预测)
                # [Batch*Time, 1, n_embd]
                last_token_emb = h_x[:, -1:, :]
                
                # 预测 Logits -> Probs -> Binary Preds
                step_logits = self.cls_head(last_token_emb) # [Batch*Time, 1, Targets]
                outputs_list.append(step_logits)
                
                # 如果不是最后一步，准备下一步的输入
                if i < self.num_windows - 1:
                    probs = torch.sigmoid(step_logits)
                    
                    # 策略: Hard Prediction (最接近原文) 
                    # 将 >0.5 的转为 1.0, 否则 0.0
                    preds = (probs > 0.5).float()
                    
                    # 也可以尝试 Soft Feedback (直接传 probs)，但这里使用 Hard 以匹配 "Explicit" 定义
                    
                    # Embed predictions: [Batch*Time, 1, n_embd]
                    # squeeze/unsqueeze 处理维度匹配 Linear 输入
                    next_input_emb = self.label_encoder(preds.squeeze(1)).unsqueeze(1)
                    
                    # Concat to input sequence
                    current_input = torch.cat([current_input, next_input_emb], dim=1)

            # 3. 拼接所有步的输出
            # [Batch*Time, Windows, Targets]
            logits_flat = torch.cat(outputs_list, dim=1)
            
            # Reshape & Permute
            logits = logits_flat.view(b, t, self.num_windows, self.num_targets).permute(0, 1, 3, 2)
            
            return SequenceClassifierOutput(loss=None, logits=logits)

    @torch.no_grad()
    def predict(self, num_x, cat_x, days, padding_mask=None):
        self.eval()
        # forward 中会自动检测 labels=None 进入自回归模式
        outputs = self(num_x, cat_x, days, labels=None, padding_mask=padding_mask)
        logits = outputs.logits
        # 取最后一个有效时间步的预测 (适配 Metrics 接口)
        if padding_mask is not None:
            last_indices = padding_mask.sum(dim=1) - 1
            last_indices = last_indices.clamp(min=0)
            # [Batch, 1, Targets, Windows]
            final_logits = logits[torch.arange(num_x.size(0)), last_indices].unsqueeze(1)
        else:
            final_logits = logits

        probs = torch.sigmoid(final_logits)
        preds = (probs > 0.5).int()
        return preds, probs