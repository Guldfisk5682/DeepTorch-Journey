import torch
from transformer import Transformer

def test_transformer():
    # 设置参数
    vocab_size = 1000
    batch_size = 4
    seq_len = 8
    d_model = 16
    num_heads = 8
    num_layers = 2
    ffn_hidden = 64
    
    # 创建模型
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_hidden=ffn_hidden
    )
    
    # 创建输入数据
    enc_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dec_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 创建掩码
    valid_len = torch.randint(1, seq_len + 1, (batch_size,))
    # 创建因果掩码，防止解码器看到未来信息
    # 首先创建一个上三角矩阵，对角线为1，上三角为0
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()   
    # 将掩码扩展到适合多头注意力的形状
    # unsqueeze(0) 在第0维增加一个维度，用于批处理
    # expand 将掩码复制到每个batch和每个注意力头
    # 最终形状为 (batch_size * num_heads, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
    
    # 前向传播
    output = model(enc_input, dec_input, valid_len, causal_mask)
    
    # 检查输出形状
    expected_shape = (batch_size, seq_len, vocab_size)
    assert output.shape == expected_shape, f"输出形状错误：期望 {expected_shape}，得到 {output.shape}"
    
    print("测试通过！输出形状正确。")
    print(f"输入形状：{enc_input.shape}")
    print(f"输出形状：{output.shape}")
    
if __name__ == "__main__":
    test_transformer() 