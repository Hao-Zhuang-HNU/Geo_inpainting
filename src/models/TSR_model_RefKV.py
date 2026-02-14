import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
from .transformer_RefKV import BlockAxial, SelfAttention, GELU
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

class EdgeLineGPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    use_ref_kv = False 
    # 保持 1024，因为 Cross Attention 是全局的
    block_size = 1024 

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# --- 新增：全局交叉注意力模块 ---
class GlobalCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, ref_feat):
        # x: [B, C, H, W] -> 需要 Flatten
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # [B, T, C]
        
        # ref_feat: [B, C, T_ref, 1] -> [B, T_ref, C]
        ref = ref_feat.squeeze(-1).transpose(1, 2)
        
        # Q 来自当前帧
        q = self.query(x_flat).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        # K, V 来自参考帧 (全局一份，无需复制！)
        k = self.key(ref).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(ref).view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Attention
        # 这里的 mask 通常不需要，因为 Ref 是全可见的
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        
        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        y = self.resid_drop(self.proj(y))
        
        # Reshape back to [B, C, H, W]
        y = y.transpose(1, 2).view(B, C, H, W)
        return y

# --- 新增：混合块 (Axial Self + Global Cross) ---
class HybridBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd) # Cross Attn 用的 LN
        
        # 1. Self-Attention: 使用 Axial (高效，负责结构连贯)
        # 注意：这里我们告诉 BlockAxial 不要处理 ref_kv
        config_copy = config # 浅拷贝
        self.self_attn = BlockAxial(config_copy) 
        
        # 2. Cross-Attention: 使用 Global (不复制内存，负责搬运参考)
        self.use_ref_kv = getattr(config, 'use_ref_kv', False)
        if self.use_ref_kv:
            self.cross_attn = GlobalCrossAttention(config)
            
        # 3. FFN
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, ref_feat=None):
        # x: [B, C, H, W]
        
        # 1. Axial Self-Attention
        # BlockAxial 内部期待 [B, C, H, W]，无需 reshape
        # 这里的 ref_feat 传 None，因为我们要在外面单独做 Cross
        x = x + self.self_attn(x, ref_feat=None) 
        
        # 2. Global Cross-Attention
        if self.use_ref_kv and ref_feat is not None:
            # 需要对 x 进行 LN 吗？通常 Pre-Norm 结构需要在 Attn 前做 LN
            # 但 x 是 4D 的，LayerNorm 默认处理最后一维
            # 我们手动处理一下维度
            x_perm = x.permute(0, 2, 3, 1) # [B, H, W, C]
            x_norm = self.ln3(x_perm).permute(0, 3, 1, 2) # Back to [B, C, H, W]
            
            # Cross Attn (Global)
            # 这里的计算量是 1024 * T_ref (256+1024)，比 Axial 的 Copy 代价小得多
            x = x + self.cross_attn(x_norm, ref_feat)

        # 3. FFN
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.ln2(x_perm)
        x_ffn = self.mlp(x_norm).permute(0, 3, 1, 2)
        x = x + x_ffn
        
        return x

class EdgeLineGPT256RelBCE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # =======================
        # 1. Shared Encoder
        # =======================
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0)
        self.act = nn.ReLU(True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        
        # Embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))
        self.type_emb_global = nn.Parameter(torch.zeros(1, 1, 256))
        self.type_emb_local  = nn.Parameter(torch.zeros(1, 1, 256))
        nn.init.normal_(self.type_emb_global, std=0.02)
        nn.init.normal_(self.type_emb_local, std=0.02)

        # Refinement
        self.ref_refinement = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )

        self.drop = nn.Dropout(config.embd_pdrop)
        
        # =======================
        # 2. Hybrid Transformer Backbone (New!)
        # =======================
        self.blocks = []
        # 使用 HybridBlock 替代原来的 BlockAxial / my_Block_2
        for _ in range(config.n_layer):
            self.blocks.append(HybridBlock(config))
        self.blocks = nn.Sequential(*self.blocks)
        
        self.ln_f = nn.LayerNorm(256)
        
        self.block_size = 32
        self.config = config
        self.use_ref_kv = getattr(config, 'use_ref_kv', False)

        # =======================
        # 3. Decoupled Decoders
        # =======================
        self.edge_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, padding=0)
        )

        self.line_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, padding=0)
        )
        
        self.act_last = nn.Sigmoid()
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        no_decay.add('pos_emb')
        no_decay.add('type_emb_global')
        no_decay.add('type_emb_local')

        for name, param in self.named_parameters():
            if 'ref_weight' in name:
                no_decay.add(name)
            elif 'ref_key' in name or 'ref_value' in name:
                decay.add(name)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def _encode(self, img_idx, edge_idx, line_idx, masks):
        img_idx = img_idx * (1 - masks)
        edge_idx = edge_idx * (1 - masks)
        line_idx = line_idx * (1 - masks)
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=1)
        
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.act(x)
        x = self.conv4(x)
        x = self.act(x)
        
        [b, c, h, w] = x.shape
        x = x.view(b, c, h * w).transpose(1, 2).contiguous()
        
        position_embeddings = self.pos_emb[:, :h * w, :]
        x = self.drop(x + position_embeddings)
        
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        return x
    
    def _process_blocks(self, x, ref_feat=None):
        """
        处理 Hybrid Blocks
        """
        for i, block in enumerate(self.blocks):
            if self.use_ref_kv and ref_feat is not None:
                # 开启 Checkpoint 以省显存
                if self.training:
                     x = checkpoint(block, x, ref_feat, use_reentrant=False)
                else:
                     x = block(x, ref_feat=ref_feat)
            else:
                x = block(x)
        return x

    def _decode(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln_f(x).permute(0, 3, 1, 2).contiguous()
        edge = self.edge_decoder(x)
        line = self.line_decoder(x)
        return edge, line

    def forward_with_logits(self, img_idx, edge_idx, line_idx, masks=None, ref_feat=None):
        x = self._encode(img_idx, edge_idx, line_idx, masks)
        x = self._process_blocks(x, ref_feat=ref_feat)
        edge, line = self._decode(x)
        return edge, line

    def extract_reference_features(self, 
                                   global_img=None, global_edge=None, global_line=None,
                                   local_img=None, local_edge=None, local_line=None, local_mask=None,
                                   local_conf=None):
        """
        Build reference features for global + local references.

        Scalar gate for local branch (residual-style):
          - global ref tokens are always kept as-is
          - local ref tokens are treated as an additional residual reference set and scaled by w
        Gate:
          w = sigmoid((conf - tau) / temp), with tau=0.30, temp=0.10
        """
        ref_list = []

        # --- 1. Global Ref (Pool 16x16) ---
        if global_img is not None or global_edge is not None:
            if global_img is None:
                B, _, H, W = global_edge.shape
                global_img = torch.zeros((B, 3, H, W), device=global_edge.device, dtype=global_edge.dtype)
            zero_mask = torch.zeros_like(global_img[:, :1, :, :])
            g_feat = self._encode(global_img, global_edge, global_line, masks=zero_mask)

            g_feat = F.adaptive_avg_pool2d(g_feat, (16, 16))
            g_feat_flat = g_feat.flatten(2).transpose(1, 2)          # [B, 256, 256]
            g_feat_final = g_feat_flat + self.type_emb_global        # [B, 256, 256]
            ref_list.append(g_feat_final)

        # --- 2. Local Ref (No Pool 32x32) + Scalar Gate ---
        if local_img is not None or local_edge is not None:
            if local_img is None:
                B, _, H, W = local_edge.shape
                local_img = torch.zeros((B, 3, H, W), device=local_edge.device, dtype=local_edge.dtype)
            else:
                B = local_img.shape[0]

            if local_mask is None:
                local_mask = torch.zeros_like(local_edge)

            l_feat = self._encode(local_img, local_edge, local_line, masks=local_mask)
            l_feat = self.ref_refinement(l_feat)

            # No Pooling
            l_feat_flat = l_feat.flatten(2).transpose(1, 2)          # [B, 1024, 256]
            l_feat_final = l_feat_flat + self.type_emb_local         # [B, 1024, 256]

            # ---- scalar gate for local branch (residual-style: scale local tokens only) ----
            if local_conf is not None:
                # recommended gate params
                tau = 0.30
                temp = 0.10

                dev = l_feat_final.device
                conf = local_conf
                if not torch.is_tensor(conf):
                    conf = torch.tensor(conf, device=dev, dtype=torch.float32)
                else:
                    conf = conf.to(device=dev, dtype=torch.float32)

                # shape normalize to [B]
                if conf.dim() == 0:
                    conf = conf.view(1).expand(B)
                else:
                    conf = conf.view(conf.shape[0])

                conf = conf.clamp(0.0, 1.0)                          # [B]
                w = torch.sigmoid((conf - tau) / temp)               # [B]
                w = w.to(dtype=l_feat_final.dtype).view(B, 1, 1)     # [B,1,1] for broadcast

                l_feat_final = l_feat_final * w

            ref_list.append(l_feat_final)

        if not ref_list:
            return None

        # concat tokens: [B, T_total, C] -> [B, C, T_total, 1]
        final_ref = torch.cat(ref_list, dim=1)
        final_ref = final_ref.permute(0, 2, 1).unsqueeze(-1)

        return final_ref


    def forward(self, img_idx, edge_idx, line_idx, edge_targets=None, line_targets=None, masks=None, ref_feat=None):
        edge, line = self.forward_with_logits(img_idx, edge_idx, line_idx, masks, ref_feat)
        loss = 0
        if edge_targets is not None and line_targets is not None:
            loss = F.binary_cross_entropy_with_logits(edge.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      edge_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                      reduction='none')
            loss = loss + F.binary_cross_entropy_with_logits(line.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             line_targets.permute(0, 2, 3, 1).contiguous().view(-1, 1),
                                                             reduction='none')
            masks_ = masks.view(-1, 1)
            loss *= masks_
            loss = torch.mean(loss)
        
        edge, line = self.act_last(edge), self.act_last(line)
        return edge, line, loss