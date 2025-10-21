import torch
import struct
import numpy as np
from pathlib import Path
from typing import Dict, Any

def parse_config(file_path: str) -> Dict[str, Any]:
    """解析模型配置头，生成与ModelArgs兼容的配置"""
    with open(file_path, 'rb') as f:
        # 读取Config结构体 - 根据实际结构体大小调整
        # 假设Config包含常见的Transformer参数
        config_data = f.read(7 * 4)  # 7个int，每个4字节
        
        # 解析Config字段（需要根据实际结构调整）
        config_values = struct.unpack('7i', config_data)
        
        # 创建与ModelArgs兼容的配置字典
        model_args = {
            'dim': config_values[0],
            'hidden_dim': config_values[1], 
            'n_layers': config_values[2],
            'n_heads': config_values[3],
            'n_kv_heads': config_values[4],
            'vocab_size': abs(config_values[5]),  # 取绝对值
            'max_seq_len': config_values[6],  # 通常叫max_seq_len而不是seq_len
            # 添加其他可能需要的参数
            'multiple_of': 256,  # 常见的FFN维度倍数
            'norm_eps': 1e-5,    # 常见的归一化epsilon
        }
        
        return model_args

def read_weights(file_path: str, model_args: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """读取权重数据并转换为与Transformer模型兼容的状态字典"""
    with open(file_path, 'rb') as f:
        # 跳过Config头部
        f.seek(7 * 4)  # 跳过7个int的Config
        
        # 读取剩余的权重数据
        weights_data = np.fromfile(f, dtype=np.float32)
    
    # 将数据转换为PyTorch张量
    weights_ptr = 0
    state_dict = {}
    
    # 根据模型结构解析权重
    dim = model_args['dim']
    hidden_dim = model_args['hidden_dim'] 
    n_layers = model_args['n_layers']
    vocab_size = model_args['vocab_size']
    n_heads = model_args['n_heads']
    n_kv_heads = model_args.get('n_kv_heads', n_heads)  # 默认为n_heads
    
    head_dim = dim // n_heads
    
    # 词嵌入权重
    tok_embeddings_size = vocab_size * dim
    state_dict['tok_embeddings.weight'] = torch.from_numpy(
        weights_data[weights_ptr:weights_ptr + tok_embeddings_size]
    ).reshape(vocab_size, dim)
    weights_ptr += tok_embeddings_size
    
    # 各Transformer层的权重
    for layer_idx in range(n_layers):
        layer_prefix = f'layers.{layer_idx}'
        
        # 注意力层归一化 (RMSNorm)
        rms_att_size = dim
        state_dict[f'{layer_prefix}.attention_norm.weight'] = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + rms_att_size]
        )
        weights_ptr += rms_att_size
        
        # 注意：实际实现中，QKV可能合并存储，这里需要根据原始代码调整
        
        # WQ权重 (多头投影)
        wq_size = dim * dim
        wq_weight = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + wq_size]
        ).reshape(dim, dim)
        state_dict[f'{layer_prefix}.attention.wq.weight'] = wq_weight
        weights_ptr += wq_size
        
        # WK权重
        wk_size = dim * (n_kv_heads * head_dim)
        wk_weight = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + wk_size]
        ).reshape(dim, n_kv_heads * head_dim)
        state_dict[f'{layer_prefix}.attention.wk.weight'] = wk_weight
        weights_ptr += wk_size
        
        # WV权重
        wv_size = dim * (n_kv_heads * head_dim)
        wv_weight = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + wv_size]
        ).reshape(dim, n_kv_heads * head_dim)
        state_dict[f'{layer_prefix}.attention.wv.weight'] = wv_weight
        weights_ptr += wv_size
        
        # WO权重 (输出投影)
        wo_size = dim * dim
        wo_weight = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + wo_size]
        ).reshape(dim, dim)
        state_dict[f'{layer_prefix}.attention.wo.weight'] = wo_weight
        weights_ptr += wo_size
        
        # FFN层归一化 (RMSNorm)
        rms_ffn_size = dim
        state_dict[f'{layer_prefix}.ffn_norm.weight'] = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + rms_ffn_size]
        )
        weights_ptr += rms_ffn_size
        
        # FFN权重 (通常使用SwiGLU或类似结构)
        # w1 (gate投影)
        w1_size = dim * hidden_dim
        state_dict[f'{layer_prefix}.feed_forward.w1.weight'] = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + w1_size]
        ).reshape(dim, hidden_dim)
        weights_ptr += w1_size
        
        # w2 (下投影)
        w2_size = hidden_dim * dim
        state_dict[f'{layer_prefix}.feed_forward.w2.weight'] = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + w2_size]
        ).reshape(hidden_dim, dim)
        weights_ptr += w2_size
        
        # w3 (上投影)
        w3_size = dim * hidden_dim
        state_dict[f'{layer_prefix}.feed_forward.w3.weight'] = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + w3_size]
        ).reshape(dim, hidden_dim)
        weights_ptr += w3_size
    
    # 最终层归一化
    rms_final_size = dim
    state_dict['norm.weight'] = torch.from_numpy(
        weights_data[weights_ptr:weights_ptr + rms_final_size]
    )
    weights_ptr += rms_final_size
    
    # 输出权重 (如果不共享权重)
    # 注意：原始代码中vocab_size的正负表示是否共享权重
    shared_weights = model_args['vocab_size'] == abs(model_args['vocab_size'])
    if not shared_weights:
        output_size = vocab_size * dim
        state_dict['output.weight'] = torch.from_numpy(
            weights_data[weights_ptr:weights_ptr + output_size]
        ).reshape(vocab_size, dim)
        weights_ptr += output_size
    else:
        # 如果共享权重，输出层使用嵌入权重
        state_dict['output.weight'] = state_dict['tok_embeddings.weight']
    
    # 验证是否读取了所有数据
    remaining_data = len(weights_data) - weights_ptr
    if remaining_data > 0:
        print(f"Warning: {remaining_data} values remaining in weight file")
    
    return state_dict

def convert_to_torch_checkpoint(checkpoint_path: str, output_path: str):
    """主转换函数 - 生成与load_checkpoint兼容的检查点"""
    print(f"Converting {checkpoint_path} to PyTorch checkpoint...")
    
    # 解析配置
    model_args = parse_config(checkpoint_path)
    print(f"Model args: {model_args}")
    
    # 读取权重
    state_dict = read_weights(checkpoint_path, model_args)
    
    # 创建与load_checkpoint兼容的检查点字典
    checkpoint_dict = {
        'model_args': model_args,
        'model': state_dict,
        # 可以添加其他元数据
        'checkpoint_version': '1.0',
        'converted_from': Path(checkpoint_path).name
    }
    
    # 保存PyTorch检查点
    torch.save(checkpoint_dict, output_path)
    
    print(f"Conversion completed! Saved to {output_path}")
    
    # 验证转换
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params:,}")
    
    # 测试加载
    try:
        # 注意：这里需要你有相应的Transformer类定义
        # 为了测试，我们可以只检查状态字典的结构
        print("Checking state dict keys:")
        for key in list(state_dict.keys())[:5]:  # 只显示前5个键
            print(f"  {key}: {state_dict[key].shape}")
        if len(state_dict) > 5:
            print(f"  ... and {len(state_dict) - 5} more keys")
    except Exception as e:
        print(f"Load test warning: {e}")

# 使用示例
if __name__ == "__main__":
    checkpoint_path = "stories110M.bin"  # 原始模型文件
    output_path = "stories110M.pt"    # 输出PyTorch检查点
    
    convert_to_torch_checkpoint(checkpoint_path, output_path)
    
    # 测试加载转换后的模型
    def test_loaded_checkpoint():
        checkpoint_dict = torch.load(output_path, map_location='cpu', weights_only=False)
        print("Loaded checkpoint keys:", checkpoint_dict.keys())
        print("Model args:", checkpoint_dict['model_args'])
        print("State dict keys sample:", list(checkpoint_dict['model'].keys())[:3])
    
    test_loaded_checkpoint()

