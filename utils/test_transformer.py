import torch
from transformer import Transformer # 假设你的文件名是 transformer.py
from transformer import create_causal_mask
from try_device import try_gpu

def test_transformer():
    # 设置参数
    vocab_size = 1000
    batch_size = 4
    seq_len = 8
    d_model = 16 # d_model 必须能被 num_heads 整除
    num_heads = 8  # 例如，d_model=16, num_heads=8, head_dim=2. 这是允许的。
    num_layers = 2
    ffn_hidden = 64

    # 确定设备
    device = try_gpu()
    print(f"Using device: {device}")
    
    # 创建模型
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_hidden=ffn_hidden
    ).to(device) # 将模型移动到设备
    
    # 创建输入数据
    # enc_input_ids 和 dec_input_ids 应该是 LongTensor
    enc_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    dec_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # 创建掩码
    # enc_padding_valid_len 是编码器输入中每个序列的实际长度 (用于padding)
    enc_padding_valid_len = torch.randint(1, seq_len + 1, (batch_size,), device=device)

    # dec_causal_mask_tensor 是解码器自注意力机制的因果掩码
    dec_causal_mask_tensor = create_causal_mask(seq_len, batch_size, num_heads, device=device)
    
    # 前向传播
    # 使用更新后的参数名称
    output = model(enc_input_ids, dec_input_ids, enc_padding_valid_len, dec_causal_mask_tensor)
    
    # 检查输出形状
    expected_shape = (batch_size, seq_len, vocab_size)
    assert output.shape == expected_shape, f"输出形状错误：期望 {expected_shape}，得到 {output.shape}"
    
    print("测试通过！输出形状正确。")
    print(f"输入形状 (enc): {enc_input_ids.shape}")
    print(f"输入形状 (dec): {dec_input_ids.shape}")
    print(f"输出形状: {output.shape}")
    
if __name__ == "__main__":
    test_transformer()