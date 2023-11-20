from llambda.configs import get_top_head_config

import torch


def ab():
    return get_top_head_config(5)

def compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10, T=-1):
    model_resid_dim = model_config['resid_dim']
    model_n_heads = model_config['n_heads']
    model_head_dim = model_resid_dim // model_n_heads
    device = model.device

    # Universal Set of Heads
    top_heads = get_top_head_config(model_config['name_or_path'])

    top_heads = top_heads[:n_top_heads]

    # Compute Function Vector as sum of influential heads
    function_vector = torch.zeros((1,1,model_resid_dim)).to(device)

    for L,H,_ in top_heads:
        if 'gpt2-xl' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.c_proj
        elif 'gpt-j' in model_config['name_or_path']:
            out_proj = model.transformer.h[L].attn.out_proj
        elif 'llama' in model_config['name_or_path'].lower():
            out_proj = model.model.layers[L].self_attn.o_proj
        elif 'gpt-neox' in model_config['name_or_path']:
            out_proj = model.gpt_neox.layers[L].attention.dense

        x = torch.zeros(model_resid_dim)
        x[H*model_head_dim:(H+1)*model_head_dim] = mean_activations[L,H,T]
        d_out = out_proj(x.reshape(1,1,model_resid_dim).to(device).to(model.dtype))

        function_vector += d_out
        function_vector = function_vector.to(model.dtype)
    function_vector = function_vector.reshape(1, model_resid_dim)

    return function_vector, top_heads
