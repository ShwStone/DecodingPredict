import torch

class AttentionProcessor:
    """
    专门负责处理 Attention 矩阵的聚合。
    当前阶段：(Layers, Heads, Seq, Seq) -> (Seq, Seq)
    """
    def __init__(self, head_mode='mean', layer_mode='mean'):
        self.head_mode = head_mode
        self.layer_mode = layer_mode

    def _aggregate_heads(self, attentions: torch.Tensor) -> torch.Tensor:
        """多头聚合: 将 Multi-Head 压缩为 Single Map"""
        # 鲁棒性检查: (Layers, 1, Heads, S, S) -> (Layers, Heads, S, S)
        if attentions.dim() == 5 and attentions.shape[1] == 1:
            attentions = attentions.squeeze(1)
        
        # (Layers, Heads, S, S) -> (Layers, S, S)
        return torch.mean(attentions, dim=1)

    def _aggregate_layers(self, attn_layer_maps: torch.Tensor) -> torch.Tensor:
        """层间聚合: 将多层 Attention Map 融合"""
        # (Layers, S, S) -> (S, S)
        return torch.mean(attn_layer_maps, dim=0)

    def compute_weights(self, global_attn_map: torch.Tensor) -> torch.Tensor:
        """
        [预留方法] 计算 Token 重要性权重。
        Input: (S, S)
        Output: (S,) 归一化后的权重
        """
        s = torch.sum(global_attn_map, dim=0) 
        weights = s / (torch.sum(s) + 1e-9)
        return weights

    def process(self, attentions: torch.Tensor) -> torch.Tensor:
        """
        处理 Attention 的主入口。
        当前直接返回聚合后的 Attention 矩阵 (Seq, Seq)，不进行权重计算。
        """
        # 1. 多头聚合
        layer_maps = self._aggregate_heads(attentions)
        # 2. 层间聚合
        global_map = self._aggregate_layers(layer_maps)
        
        return global_map


class HiddenStateProcessor:
    """
    专门负责处理 Hidden States 的特征提取。
    当前阶段：(Layers, Seq, Dim) -> (Dim,) 使用纯平均聚合。
    """
    def __init__(self, layer_mode='mean', token_mode='mean'):
        self.layer_mode = layer_mode
        self.token_mode = token_mode

    def _aggregate_layers(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """层间聚合: (Layers, S, Dim) -> (S, Dim)"""
        if hidden_states.dim() == 4 and hidden_states.shape[1] == 1:
            hidden_states = hidden_states.squeeze(1)
        
        if self.layer_mode == 'mean':
            return torch.mean(hidden_states, dim=0)
        elif self.layer_mode == 'last':
            return hidden_states[-1]
        else:
            raise ValueError(f"Unknown layer_mode: {self.layer_mode}")

    def _aggregate_tokens(self, hidden_seq: torch.Tensor) -> torch.Tensor:
        """
        Token 间聚合: (S, Dim) -> (Dim,)
        当前仅实现平均聚合 (Mean Pooling)。
        """
        return torch.mean(hidden_seq, dim=0)

    def process(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        处理 Hidden State 的主入口。
        """
        # 1. 层间聚合
        h_seq = self._aggregate_layers(hidden_states)
        
        # 2. Token 间聚合 (平均)
        final_feature = self._aggregate_tokens(h_seq)
        
        return final_feature


class FeatureEngineer:
    """
    总控类。
    """
    def __init__(self):
        self.attn_processor = AttentionProcessor()
        self.hidden_processor = HiddenStateProcessor()

    def process(self, hidden_states: torch.Tensor, attentions: torch.Tensor = None, kv_cache=None) -> torch.Tensor:
        """
        外部调用的统一接口。
        当前阶段：仅使用 Hidden State 进行特征提取。
        """
        # 暂时忽略 attentions 和 kv_cache，仅跑 Hidden State Baseline
        final_feature = self.hidden_processor.process(hidden_states)
        
        return final_feature