import logging
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

# --- 1. 普通自注意力 (用于 Axial Block, 双向) ---
class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, use_ref_kv=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.use_ref_kv = use_ref_kv
        
        # Reference 投影层
        if use_ref_kv:
            self.ref_key = nn.Linear(n_embd, n_embd)
            self.ref_value = nn.Linear(n_embd, n_embd)
            # 零初始化：保证初始状态下不干扰主干流
            nn.init.zeros_(self.ref_key.weight)
            nn.init.zeros_(self.ref_key.bias)
            nn.init.zeros_(self.ref_value.weight)
            nn.init.zeros_(self.ref_value.bias)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None, rel_pos=None, ref_feat=None):
        B, T, C = x.size()
        
        # 1. Self QKV
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 2. Ref KV Concatenation
        if ref_feat is not None and self.use_ref_kv:
            B_ref, T_ref, C_ref = ref_feat.size()
            ref_k = self.ref_key(ref_feat).view(B_ref, T_ref, self.n_head, C // self.n_head).transpose(1, 2)
            ref_v = self.ref_value(ref_feat).view(B_ref, T_ref, self.n_head, C // self.n_head).transpose(1, 2)
            k = torch.cat([k, ref_k], dim=2)
            v = torch.cat([v, ref_v], dim=2)

        # 3. Masking Logic
        # 如果提供了 mask (通常是 padding mask), 扩展它以适配 Ref
        attn_bias = None
        if mask is not None:
            # mask: [B, 1, 1, T], 1=Masked, 0=Visible
            if ref_feat is not None and self.use_ref_kv:
                # Ref 部分全部可见 (pad 0)
                mask_pad = F.pad(mask, (0, T_ref), value=0) 
                attn_mask_bool = (mask_pad == 1)
            else:
                attn_mask_bool = (mask == 1)
            
            attn_bias = torch.zeros_like(attn_mask_bool, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(attn_mask_bool, float('-inf'))

        # 4. Relative Position Logic
        # rel_pos 只加在 Self 部分 (前 T 个位置)
        if rel_pos is not None:
            if attn_bias is None:
                T_total = k.size(2)
                attn_bias = torch.zeros((B, 1, T, T_total), device=q.device, dtype=q.dtype)
            
            # Expand headers: [B, H, T, T_total]
            attn_bias = attn_bias.expand(-1, self.n_head, -1, -1).clone()
            
            # 只叠加到 Self 区域
            attn_bias[:, :, :, :T] = attn_bias[:, :, :, :T] + rel_pos

        # 5. Attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop.p if self.training else 0.0)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

# --- 2. 轴向注意力 (Axial Block) ---
class AxialAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H, W,
                 add_rel_pos=True, use_ref_kv=False):
        super().__init__()
        self.rln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.cln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
        
        # 内部使用 SelfAttention
        self.attn_row = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, use_ref_kv=use_ref_kv)
        self.attn_col = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, use_ref_kv=use_ref_kv)
        
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )
        self.add_rel_pos = add_rel_pos
        self.row_rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
        self.col_rel_pos_bias = nn.Linear(2 * W - 1, n_head, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
        T = hidden_states.shape[1]
        position_ids = torch.arange(T, dtype=torch.long).unsqueeze(0)
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        if row:
            rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        return rel_pos.contiguous()

    def forward(self, x, ref_feat=None):
        [b, c, h, w] = x.shape
        x0 = x.clone()
        x0 = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # --- Row Attention ---
        x = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        
        ref_row = None
        if ref_feat is not None:
            # Expand Ref for Row batching: [B, T_ref, C] -> [B*W, T_ref, C]
            rf = ref_feat.squeeze(-1).permute(0, 2, 1)
            ref_row = rf.unsqueeze(1).expand(-1, w, -1, -1).reshape(b * w, -1, c)

        row_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=h, row=True) if self.add_rel_pos else None
        x_row = self.attn_row(self.rln1(x), mask=None, rel_pos=row_rel_pos, ref_feat=ref_row)
        x_row = x_row.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)

        # --- Col Attention ---
        x = x.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b * h, w, c)
        
        ref_col = None
        if ref_feat is not None:
            # Expand Ref for Col batching: [B*H, T_ref, C]
            rf = ref_feat.squeeze(-1).permute(0, 2, 1)
            ref_col = rf.unsqueeze(1).expand(-1, h, -1, -1).reshape(b * h, -1, c)

        col_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=w, row=False) if self.add_rel_pos else None
        x_col = self.attn_col(self.cln1(x), mask=None, rel_pos=col_rel_pos, ref_feat=ref_col)
        x_col = x_col.reshape(b, h, w, c).reshape(b, h * w, c)

        # --- FFN ---
        x = x0 + x_row + x_col
        x = x + self.ff(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x

class BlockAxial(AxialAttention):
    def __init__(self, config):
        super().__init__(config.n_embd, config.n_head, config.attn_pdrop, config.resid_pdrop, 32, 32,
                         use_ref_kv=getattr(config, 'use_ref_kv', False))

# --- 3. 因果自注意力 (用于 Sequence Block) ---
# 【修复】：之前漏掉了这个类，现在补上
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        self.use_ref_kv = getattr(config, 'use_ref_kv', False)
        if self.use_ref_kv:
            self.ref_key = nn.Linear(config.n_embd, config.n_embd)
            self.ref_value = nn.Linear(config.n_embd, config.n_embd)
            nn.init.zeros_(self.ref_key.weight)
            nn.init.zeros_(self.ref_key.bias)
            nn.init.zeros_(self.ref_value.weight)
            nn.init.zeros_(self.ref_value.bias)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        # 因果掩码：下三角矩阵
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, ref_feat=None):
        B, T, C = x.size()
        
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if ref_feat is not None and self.use_ref_kv:
            B_ref, T_ref, C_ref = ref_feat.size()
            ref_k = self.ref_key(ref_feat).view(B_ref, T_ref, self.n_head, C // self.n_head).transpose(1, 2)
            ref_v = self.ref_value(ref_feat).view(B_ref, T_ref, self.n_head, C // self.n_head).transpose(1, 2)
            k = torch.cat([k, ref_k], dim=2)
            v = torch.cat([v, ref_v], dim=2)

        # 构造掩码
        # Self 部分：因果掩码 (Current Mask)
        # Ref 部分：完全可见 (Ref Mask)
        causal_mask = self.mask[:, :, :T, :T] 
        
        if ref_feat is not None and self.use_ref_kv:
            # Ref Mask: 全 1 (可见)
            T_ref = ref_feat.size(1)
            ref_mask = torch.ones(1, 1, T, T_ref, device=x.device)
            full_mask = torch.cat([causal_mask, ref_mask], dim=3)
            
            # 生成 Attention Bias (0 for visible, -inf for masked)
            # mask=0 表示不可见，mask=1 表示可见
            attn_bias = torch.zeros((1, 1, T, T + T_ref), device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(full_mask == 0, float('-inf'))
        else:
            attn_bias = torch.zeros((1, 1, T, T), device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(causal_mask == 0, float('-inf'))

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop.p if self.training else 0.0)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

# --- 4. 混合块 (Sequence Block) ---
class my_Block_2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        # 现在 CausalSelfAttention 已经定义了，不会报错
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU2(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
        self.use_ref_kv = getattr(config, 'use_ref_kv', False)

    def forward(self, x, ref_feat=None):
        [b, c, h, w] = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        
        rf = None
        if self.use_ref_kv and ref_feat is not None:
            # ref_feat: [B, C, T_ref, 1] -> [B, T_ref, C]
            rf = ref_feat.squeeze(-1).permute(0, 2, 1)

        x_flat = x_flat + self.attn(self.ln1(x_flat), ref_feat=rf)
        x_flat = x_flat + self.mlp(self.ln2(x_flat))
        
        x = x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return x