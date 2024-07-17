import argparse
import os
from functools import partial

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, utils

from attn_sae import *
from sae_training.sae_group import SAEGroup
from sae_training.utils import LMSparseAutoencoderSessionloader

from e2e_sae import SAETransformer

from error_eval import cos_sim

def error_extrapolation(activation, hook, sae_out, length, pos=None):
    error_vector = (sae_out - activation)
    error_unit_vector = error_vector/error_vector.norm(dim=-1, keepdim=True)

    perturbed_activation = activation + error_unit_vector * length

    if pos is None:
        activation[:] = perturbed_activation
    else:
        activation[:, pos] = perturbed_activation[:, pos]
        
    return activation

def create_ablation_hooks(sae_out, pos=None):
    ablation_hooks = []
    for length in range(1, 211, 10):
        ablation_hooks.append((f'error_length_{length}', 
                               partial(error_extrapolation, sae_out=sae_out, length = length, pos=pos)))
    return ablation_hooks

def run_all_ablations(model, batch_tokens, ablation_hooks, layer, hook_loc="resid_pre"):
    
    orginal_logits = model(batch_tokens)
    
    batch_size, seq_len = batch_tokens.shape
    batch_result_df = pd.DataFrame({
        "token": batch_tokens[:, :-1].flatten().cpu().numpy(),
        "position": einops.repeat(
            np.arange(seq_len), "seq -> batch seq", batch=batch_size)[:, :-1].flatten(),
        "loss": utils.lm_cross_entropy_loss(
            orginal_logits, batch_tokens, per_token=True).flatten().cpu().numpy(),
    })
    
    original_log_probs = orginal_logits.log_softmax(dim=-1)
    del orginal_logits
    
    for hook_name, hook in ablation_hooks:
        
        intervention_logits = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(utils.get_act_name(hook_loc, layer), hook)]
        )
        
        intervention_loss = utils.lm_cross_entropy_loss(
            intervention_logits, batch_tokens, per_token=True
        )#.flatten().cpu().numpy()
        
        intervention_log_probs = intervention_logits.log_softmax(dim=-1)
        
        intervention_kl_div = F.kl_div(
            intervention_log_probs, 
            original_log_probs,
            log_target=True, 
            reduction='none'
        ).sum(dim=-1)
        
        batch_result_df[hook_name + "_loss"] = intervention_loss.flatten().cpu().numpy()
        batch_result_df[hook_name + "_kl"] = intervention_kl_div[:, :-1].flatten().cpu().numpy()
    
    return batch_result_df


def load_sae(layer):
    REPO_ID = "jbloom/GPT2-Small-SAEs"
    FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    model, sparse_autoencoder, _ = (
        LMSparseAutoencoderSessionloader.load_session_from_pretrained(path=path)
    )
    
    sae_group = SAEGroup(sparse_autoencoder['cfg'])

    sae = sae_group.autoencoders[0]
    
    sae.load_state_dict(sparse_autoencoder['state_dict'])
    
    sae.eval() 
    
    return sae, model


def load_attn_sae(layer):
    auto_encoder_names = {
        0 : "gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        1 : "gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5",
        2 : "gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4",
        3 : "gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        4 : "gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7",
        5 : "gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        6 : "gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        7 : "gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        8 : "gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6",
        9 : "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9",
        10 : "gpt2-small_L10_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v9",
        11 : "gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9"
    }
    auto_encoder_run = auto_encoder_names[layer]
    encoder = AutoEncoder.load_from_hf(auto_encoder_run, hf_repo="ckkissane/attn-saes-gpt2-small-all-layers")
    model = HookedTransformer.from_pretrained(encoder.cfg["model_name"]).to(DTYPES[encoder.cfg["enc_dtype"]]).to(encoder.cfg["device"])
    
    return encoder, model


def run_error_eval_experiment(sae, model, token_tensor, layer, batch_size=64, pos=None, hook_loc="resid_pre", e2e = False):
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    dataloader = torch.utils.data.DataLoader(
        token_tensor,
        batch_size=batch_size,
        shuffle=False
    )
    
    activation_loc = utils.get_act_name(hook_loc, layer)

    result_dfs = []
    for ix, batch_tokens in enumerate(tqdm.tqdm(dataloader)):
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                batch_tokens, 
                prepend_bos=True,
                names_filter=[activation_loc]
            )
            activations = cache[activation_loc]
            if hook_loc == "z":
                activations = einops.rearrange(
                    activations, "batch seq n_heads d_head -> batch seq (n_heads d_head)",
                )

            
            # for E2E SAEs
            if e2e:
                sae_out, feature_acts = sae(activations)
            else:
                sae_out, feature_acts, _, _, _, _ = sae(activations)
            
            ablation_hooks = create_ablation_hooks(sae_out, pos=pos)
            
            if hook_loc == "z":
                ablation_hooks = [
                    (name, partial(attn_hook_wrapper, hook_fn=hook_fn))
                    for name, hook_fn in ablation_hooks
                ]
            
            batch_result_df = run_all_ablations(model, batch_tokens, ablation_hooks, layer=layer, hook_loc=hook_loc)
            
            l0 = (feature_acts > 0).float().sum(dim=-1).cpu().numpy()[:, :-1].flatten()
            l1 = feature_acts.abs().sum(dim=-1).cpu().numpy()[:, :-1].flatten()
            reconstruction_error = (activations - sae_out).norm(dim=-1).cpu().numpy()[:, :-1].flatten()
            
            batch_result_df['sae_l0'] = l0
            batch_result_df['sae_l1'] = l1
            batch_result_df['reconstruction_error'] = reconstruction_error
            batch_result_df['norm'] = activations.norm(dim=-1).cpu().numpy()[:, :-1].flatten()
            batch_result_df['sae_norm'] = sae_out.norm(dim=-1).cpu().numpy()[:, :-1].flatten()
            batch_result_df['cos'] = cos_sim(activations, sae_out).cpu().numpy()[:, :-1].flatten()
            
            result_dfs.append(batch_result_df)
            
    return pd.concat(result_dfs).reset_index(drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # layer, batchsize, num_batches, output_dir, pos 
    parser.add_argument("--hook_loc", type=str, default="resid_pre")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="error_extrapolation_results")
    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--e2e", type=str, default=None)
    
    args = parser.parse_args()

    print("loading sae and model")

    if args.e2e is None:
        e2e_tag = False
        if args.hook_loc == "resid_pre":
            sae, model = load_sae(args.layer)
        elif args.hook_loc == "z":
            sae, model = load_attn_sae(args.layer)
        else:
            raise ValueError(f"Unsupported hook location {args.hook_loc}")
    else:
        e2e_tag = True
        model_id = args.e2e
        text = "sparsify/gpt2/" + model_id
        sae_transformer_model = SAETransformer.from_wandb(text)

        model = sae_transformer_model.tlens_model
        sae = sae_transformer_model.saes[f"blocks-{args.layer}-hook_{args.hook_loc}"]

    sae = sae.to(args.device)
    model = model.to(args.device)

    print("loading token tensors")
    
    token_tensor = torch.load("../token_tensor.pt").to(args.device)

    print("finished loading token tensors")
    
    if args.repeat > 1:
        token_tensor = token_tensor[: args.batch_size]
        token_tensor = einops.repeat(
            token_tensor, 
            "batch seq -> (repeat batch) seq", 
            repeat=args.repeat
        )
    
    result_df = run_error_eval_experiment(
        sae, 
        model, 
        token_tensor, 
        args.layer, 
        args.batch_size, 
        args.pos, 
        args.hook_loc,
        args.e2e
    )
    
    save_path = os.path.join("../results/" + args.output_dir, f"gpt2_{args.hook_loc}")
    os.makedirs(save_path, exist_ok=True)
    pos_label = 'all' if args.pos is None else args.pos
    
    save_name = f"layer_{args.layer}_pos_{pos_label}.csv"

    if args.e2e is not None:
        save_name = f"e2e_{args.e2e}_" + save_name
    
    result_df.to_csv(os.path.join(save_path, save_name), index=False)
