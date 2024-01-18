import argparse
import os
from collections import OrderedDict

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import accelerate

transformer_layer_name_list = {
    "input_norm": [
        "input_norm.weight",
        "self_attention.norm_qkv.layer_norm_weight",
    ],
    "query_key_value": [
        "self_attention.query_key_value.weight",
        "self_attention.norm_qkv.weight",
    ],
    "query": ["self_attention.query.weight"],
    "key_value": ["self_attention.key_value.weight"],
    "o_proj": ["self_attention.dense.weight", "self_attention.proj.weight"],
    "mlp_gate_up": ["mlp.dense_h_to_4h.weight", "norm_mlp.fc1_weight"],
    "mlp_down": ["mlp.dense_4h_to_h.weight", "norm_mlp.fc2_weight"],
    "post_attention_norm": [
        "post_attention_norm.weight",
        "norm_mlp.layer_norm_weight",
    ],
}


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def get(dicts, key):
    return [dict[key] for dict in dicts]


def check_get(dicts, prefix, key_list):
    return [
        dict[prefix + key] for dict in dicts for key in key_list if prefix + key in dict
    ]


def check_assign(encoder, this_layer_index, this_encoder, layer_index, key_list):
    for key in key_list:
        full_key = f"layers.{layer_index}." + key
        if full_key in this_encoder:
            encoder[f"layers.{this_layer_index}." + key] = this_encoder[full_key]
            break
    return encoder


def merge_col(tensors):
    return torch.cat(
        [
            tensor["weight"] if type(tensor) is OrderedDict else tensor
            for tensor in tensors
        ],
        dim=0,
    )


def merge_row(tensors):
    return torch.cat(
        [
            tensor["weight"] if type(tensor) is OrderedDict else tensor
            for tensor in tensors
        ],
        dim=1,
    )


def convert_megatron_checkpoint(hf_model, state_dicts, model_config: LlamaConfig):
    # The model.
    models = get(state_dicts, "model")

    # The language model.
    lms = get(models, "language_model")

    # The embeddings.
    embeddings = get(lms, "embedding")

    # The word embeddings.
    word_embeddings = get(embeddings, "word_embeddings")

    # Truncate the embedding table to vocab_size rows.
    merged_padded_word_embeddings = merge_col(word_embeddings)
    merged_word_embeddings = merged_padded_word_embeddings[: model_config.vocab_size, :]
    hf_model.model.embed_tokens.load_state_dict(
        {"weight": merged_word_embeddings}, strict=True
    )

    # The transformer.
    transformers = get(lms, "encoder")

    for i in range(model_config.num_hidden_layers):
        print("Converting layer", i)
        prefix = f"layers.{i}."
        layer: LlamaDecoderLayer = hf_model.model.layers[i]

        layer.input_layernorm.load_state_dict(
            {
                "weight": check_get(
                    transformers, prefix, transformer_layer_name_list["input_norm"]
                )[0]
            },
            strict=True,
        )

        hidden_size = model_config.hidden_size
        inter_size = model_config.intermediate_size
        num_heads = model_config.num_attention_heads
        kv_heads = model_config.num_key_value_heads
        kv_hidden_size = hidden_size // num_heads * kv_heads
        if num_heads == kv_heads:
            qkv = merge_col(
                check_get(
                    transformers, prefix, transformer_layer_name_list["query_key_value"]
                )
            )
            qkv = qkv.view(num_heads, 3, hidden_size // num_heads, hidden_size)
            q, k, v = torch.chunk(qkv, 3, dim=1)
            q, k, v = (
                q.reshape(hidden_size, hidden_size),
                k.reshape(hidden_size, hidden_size),
                v.reshape(hidden_size, hidden_size),
            )
        else: 
            qkv = merge_col(
                check_get(
                    transformers, prefix, transformer_layer_name_list["query_key_value"]
                )
            )
 
            num_queries_per_key_value = num_heads // kv_heads
            qkv = qkv.view(
                kv_heads,
                num_queries_per_key_value + 2,
                hidden_size // num_heads,
                hidden_size,
            )
            q, k, v = torch.split(qkv, [num_queries_per_key_value, 1, 1], dim=1)
                 
            
            q, k, v = (
                q.reshape(hidden_size, hidden_size),
                k.reshape(kv_hidden_size, hidden_size),
                v.reshape(kv_hidden_size, hidden_size),
            )

        layer.self_attn.q_proj.load_state_dict({"weight": q}, strict=True)
        layer.self_attn.k_proj.load_state_dict({"weight": k}, strict=True)
        layer.self_attn.v_proj.load_state_dict({"weight": v}, strict=True)

        layer.self_attn.o_proj.load_state_dict(
            {
                "weight": merge_row(
                    check_get(
                        transformers, prefix, transformer_layer_name_list["o_proj"]
                    )
                )
            },
            strict=True,
        )

        gate, up = (
            merge_col(
                check_get(
                    transformers, prefix, transformer_layer_name_list["mlp_gate_up"]
                )
            )
            .view(len(state_dicts), 2, -1, hidden_size)
            .chunk(2, dim=1)
        )
        gate, up = gate.reshape(inter_size, hidden_size), up.reshape(
            inter_size, hidden_size
        )
        layer.mlp.gate_proj.load_state_dict({"weight": gate}, strict=True)
        layer.mlp.up_proj.load_state_dict({"weight": up}, strict=True)
        layer.mlp.down_proj.load_state_dict(
            {
                "weight": merge_row(
                    check_get(
                        transformers, prefix, transformer_layer_name_list["mlp_down"]
                    )
                )
            },
            strict=True,
        )

        layer.post_attention_layernorm.load_state_dict(
            {
                "weight": check_get(
                    transformers,
                    prefix,
                    transformer_layer_name_list["post_attention_norm"],
                )[0]
            },
            strict=True,
        )

    # The final norm.
    hf_model.model.norm.load_state_dict(
        {"weight": transformers[0]["final_norm.weight"]}, strict=True
    )

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_layers = get(lms, "output_layer")
    merged_padded_output_layers = merge_col(output_layers)
    merged_output_layers = merged_padded_output_layers[: model_config.vocab_size, :]
    hf_model.lm_head.load_state_dict({"weight": merged_output_layers}, strict=True)


def check_padded_vocab_size(train_args, orig_vocab_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = (
        train_args.make_vocab_size_divisible_by * train_args.tensor_model_parallel_size
    )
    while (after % multiple) != 0:
        after += 1
    assert (
        train_args.padded_vocab_size == after
    ), "Mismatched vocab size and padded vocab size."


def get_train_args(state_dict):
    args = state_dict.get("args", None)
    assert args is not None
    return args


def get_model_config(train_args, vocab_size):
    config = LlamaConfig()
    check_padded_vocab_size(train_args, vocab_size)
    config.vocab_size = vocab_size
    # config.vocab_size = train_args.padded_vocab_size
    config.max_position_embeddings = train_args.max_position_embeddings
    config.hidden_size = train_args.hidden_size
    config.num_hidden_layers = train_args.num_layers
    config.num_attention_heads = train_args.num_attention_heads
    config.num_key_value_heads = train_args.num_query_groups
    config.intermediate_size = train_args.ffn_hidden_size
    if hasattr(train_args, "rope_base"):
        config.rope_theta = train_args.rope_base
    config.pad_token_id = 0
    config.torch_dtype  = train_args.params_dtype
    return config


def load_state_dicts(input_dir):
    state_dicts = [
        torch.load(os.path.join(f.path, "model_optim_rng.pt"), map_location="cpu")
        for f in os.scandir(input_dir)
        if f.is_dir()
    ]
    args = get_train_args(state_dicts[0])
    if args.transformer_pipeline_model_parallel_size == 1:
        return state_dicts, args

    state_dicts = []
    tp_size = args.tensor_model_parallel_size
    pp_size = args.transformer_pipeline_model_parallel_size
    num_layers_per_pile = args.num_layers // pp_size
    for tp_index in range(tp_size):
        model_file = f"{input_dir}/mp_rank_{tp_index:02d}_000/model_optim_rng.pt"
        print(f"loading {model_file}")
        state_dict = torch.load(
            model_file,
            map_location="cpu",
        )
        lm = state_dict["model"]["language_model"]
        encoder = lm["encoder"]
        for pp_index in range(1, pp_size):
            model_file = f"{input_dir}/mp_rank_{tp_index:02d}_{pp_index:03d}/model_optim_rng.pt"
            this_state_dict = torch.load(
                model_file,
                map_location="cpu",
            )
            print(f"loading {model_file}")
            this_lm = this_state_dict["model"]["language_model"]
            this_encoder = this_lm["encoder"]

            if pp_index == pp_size - 1:
                lm["output_layer"] = this_lm["output_layer"]
                encoder["final_norm.weight"] = this_encoder[
                    "final_norm.weight"
                ]

            for layer_index in range(num_layers_per_pile):
                this_layer_index = layer_index + num_layers_per_pile * pp_index
                if args.num_attention_heads == args.num_query_groups:
                    encoder = check_assign(
                        encoder,
                        this_layer_index,
                        this_encoder,
                        layer_index,
                        key_list=transformer_layer_name_list["query_key_value"],
                    )
                else:
                    for key in ["query", "key_value", "query_key_value"]:
                        encoder = check_assign(
                            encoder,
                            this_layer_index,
                            this_encoder,
                            layer_index,
                            key_list=transformer_layer_name_list[key],
                        )
                for key in transformer_layer_name_list.keys():
                    if key not in ("query_key_value", "query", "key_value"):
                        encoder = check_assign(
                            encoder,
                            this_layer_index,
                            this_encoder,
                            layer_index,
                            key_list=transformer_layer_name_list[key],
                        )
        state_dicts.append(state_dict)

    return state_dicts, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to the megatron checkpoint dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the huggingface checkpoint dir",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=64000,
        help="unpadded tokenizer vocab size",
    )
    args = parser.parse_args()

    print("Load megatron checkpoint")
    state_dicts, train_args = load_state_dicts(args.input_dir)

    model_config = get_model_config(train_args, args.vocab_size)
    print(f"Model config: {model_config}", flush=True)
    
    
    print("Create hf model", flush=True)
    # with accelerate.init_empty_weights():
    hf_model = LlamaForCausalLM(model_config)
    hf_model = hf_model.to(torch.bfloat16)
    
    print("convert megatron to hf", flush=True)
    convert_megatron_checkpoint(hf_model, state_dicts, model_config)

    print("save hf model", flush=True)
    hf_model.save_pretrained(args.output_dir, safe_serialization=False)


if __name__ == "__main__":
    main()
