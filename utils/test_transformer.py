import torch
from transformer import Transformer
from transformer import create_causal_mask

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

    causal_mask=create_causal_mask(seq_len,batch_size,num_heads)
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