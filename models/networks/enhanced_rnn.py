import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedRNNEncoder(nn.Module):
    '''
    增强型RNN编码器，结合GRU、自注意力和残差连接
    可以作为LSTM的高效平替方案，保持相同的输入输出接口
    '''
    def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=True, 
                 num_layers=3, dropout=0.3, use_attention=True, use_layer_norm=True,
                 use_batch_norm=True, residual_connections=True):
        super(EnhancedRNNEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embd_method = embd_method
        self.use_attention = use_attention
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
        self.residual_connections = residual_connections
        
        # 输入投影层
        self.input_proj = nn.Linear(input_size, hidden_size)
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)
        
        # GRU层 (比LSTM更轻量且效果相当)
        self.rnn = nn.GRU(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 计算实际输出维度
        self.rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # 层归一化 (提高训练稳定性)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.rnn_output_size)
        
        # 自注意力机制
        if use_attention:
            # 多头注意力机制
            self.self_attn = nn.MultiheadAttention(
                embed_dim=self.rnn_output_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            
        # 输出投影层 (确保输出维度与LSTM一致)
        self.output_proj = nn.Sequential(
            nn.Linear(self.rnn_output_size, hidden_size),
            nn.Dropout(dropout),
            nn.GELU()
        )
        
        # 根据嵌入方法初始化相应组件
        assert embd_method in ['maxpool', 'attention', 'last', 'dense']
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)
            
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential(
                nn.Linear(self.rnn_output_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型参数"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    if 'rnn' in name:
                        nn.init.orthogonal_(param)  # RNN权重使用正交初始化
                    else:
                        nn.init.xavier_uniform_(param)  # 线性层使用Xavier初始化
                else:
                    nn.init.uniform_(param, -0.1, 0.1)  # 对1维权重使用均匀分布初始化
            elif 'bias' in name:
                nn.init.zeros_(param)  # 偏置初始化为0
                
        # 特殊处理attention权重
        if self.embd_method == 'attention' and hasattr(self, 'attention_vector_weight'):
            if len(self.attention_vector_weight.shape) >= 2:
                nn.init.xavier_uniform_(self.attention_vector_weight)
            else:
                nn.init.uniform_(self.attention_vector_weight, -0.1, 0.1)
    
    def apply_self_attention(self, x):
        """应用多头自注意力机制"""
        attn_output, _ = self.self_attn(x, x, x)
        
        # 残差连接
        if self.residual_connections:
            return x + attn_output
        return attn_output

    def embd_attention(self, r_out, h_n):
        """与原LSTM相同的注意力嵌入方法"""
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)     # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        attended_r_out = r_out * atten_weight                           # [batch_size, seq_len, hidden_size]
        return attended_r_out

    def embd_maxpool(self, r_out, h_n):
        """与原LSTM相同的最大池化嵌入方法"""
        pooled_out, _ = torch.max(r_out, dim=1, keepdim=True)  # [batch_size, 1, hidden_size]
        return pooled_out.expand_as(r_out)  # [batch_size, seq_len, hidden_size]

    def embd_last(self, r_out, h_n):
        """与原LSTM相同的last嵌入方法"""
        return r_out  # [batch_size, seq_len, hidden_size]

    def embd_dense(self, r_out, h_n):
        """与原LSTM相同的dense嵌入方法"""
        r_out_flat = r_out.view(-1, r_out.size(2))  # [batch_size * seq_len, hidden_size]
        dense_out = self.dense_layer(r_out_flat)
        return dense_out.view(r_out.size(0), r_out.size(1), self.hidden_size)  # [batch_size, seq_len, hidden_size]

    def forward(self, x):
        """前向传播"""
        batch_size, seq_len, _ = x.size()
        
        # 批归一化处理输入
        if self.use_batch_norm:
            x_reshaped = x.reshape(-1, self.input_size)
            x_bn = self.input_bn(x_reshaped)
            x = x_bn.reshape(batch_size, seq_len, self.input_size)
        
        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_size]
        
        # 保存原始输入用于残差连接
        orig_x = x
        
        # GRU处理
        r_out, h_n = self.rnn(x)  # r_out: [batch_size, seq_len, hidden_size*num_directions]
        
        # 层归一化
        if self.use_layer_norm:
            r_out = self.layer_norm(r_out)
        
        # 自注意力机制
        if self.use_attention:
            r_out = self.apply_self_attention(r_out)
        
        # 输出投影
        r_out = self.output_proj(r_out)  # 确保输出维度一致
        
        # 残差连接 (如果输入和输出维度相同)
        if self.residual_connections and orig_x.size(-1) == r_out.size(-1):
            r_out = r_out + orig_x
        
        # 应用嵌入方法
        embd = getattr(self, 'embd_' + self.embd_method)(r_out, h_n)
        
        return embd