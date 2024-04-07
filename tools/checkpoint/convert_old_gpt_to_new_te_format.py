import torch
import argparse


local_to_te_model_param_state_dict_map = {
    "input_layernorm.weight": "self_attention.linear_qkv.layer_norm_weight",
    "self_attention.query_key_value.weight": "self_attention.linear_qkv.weight",
    "self_attention.dense.weight": "self_attention.linear_proj.weight",
    "post_attention_layernorm.weight": "mlp.linear_fc1.layer_norm_weight",
    "mlp.dense_h_to_4h.weight": "mlp.linear_fc1.weight",
    "mlp.dense_4h_to_h.weight": "mlp.linear_fc2.weight",
}
        
def map_state_dict(model_state_dict, state_dict_map):
    if state_dict_map is not None:
        new_model_state_dict = {}
        if 'embedding' in model_state_dict['language_model']:
            new_model_state_dict['embedding.word_embeddings.weight'] = model_state_dict['language_model']['embedding']['word_embeddings']['weight']
        else:
            new_model_state_dict['decoder.final_layernorm.weight'] = model_state_dict['language_model']['encoder']['final_layernorm.weight']
            new_model_state_dict['output_layer.weight'] = model_state_dict['language_model']['output_layer']['weight']
        layers = model_state_dict['language_model']['encoder']
        for k, v in layers.items():
            if '_extra_state' in k:
                # not load _extra_state
                # print(f"> ignore _extra_state {k} when checkpointing")
                continue
            for old_name, new_name in state_dict_map.items():
                if old_name in k:
                    k = k.replace(old_name, new_name)
                    break
            new_model_state_dict[f"decoder.{k}"] = v
    return new_model_state_dict


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
    