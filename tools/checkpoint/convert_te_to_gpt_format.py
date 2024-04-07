import torch
import argparse


local_to_te_model_param_state_dict_map = {
    "input_layernorm.weight": "self_attention.layernorm_qkv.layer_norm_weight",
    "self_attention.query_key_value.weight": "self_attention.layernorm_qkv.weight",
    "self_attention.dense.weight": "self_attention.proj.weight",
    "post_attention_layernorm.weight": "layernorm_mlp.layer_norm_weight",
    "mlp.dense_h_to_4h.weight": "layernorm_mlp.fc1_weight",
    "mlp.dense_4h_to_h.weight": "layernorm_mlp.fc2_weight",
}


local_to_te_model_param_state_dict_map = dict([(v, k) for k, v in local_to_te_model_param_state_dict_map.items()])
        
def map_state_dict(model_state_dict, state_dict_map):
    if state_dict_map is not None:
        layers = model_state_dict['language_model']['encoder']
        new_layers = {}
        for k, v in layers.items():
            if '_extra_state' in k:
                # not load _extra_state
                # print(f"> ignore _extra_state {k} when checkpointing")
                continue
            for old_name, new_name in state_dict_map.items():
                if old_name in k:
                    k = k.replace(old_name, new_name)
                    break
            new_layers[k] = v
        model_state_dict['language_model']['encoder'] = new_layers
        del layers
    return model_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="transpose Megatron GPT model to trnasformer engine format")
    parser.add_argument('--in-file', type=str, required=True, help='input file of model pt')
    parser.add_argument('--out-file', type=str, required=True, help='output file of model pt')
    args = parser.parse_args()
    
    print(f"read file {args.in_file}")
    state_dict = torch.load(args.in_file, "cpu")
    
    model_state_dict = state_dict['model']
    te_model_state_dict = map_state_dict(model_state_dict, local_to_te_model_param_state_dict_map)
    
    state_dict['model'] = te_model_state_dict
    print(f"saving to {args.out_file}")
    torch.save(state_dict, args.out_file)
    
