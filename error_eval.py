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


def cos_sim(a, b):
    return einops.einsum(
        a, 
        b, 
        "batch seq dim, batch seq dim -> batch seq"
    ) / (a.norm(dim=-1) * b.norm(dim=-1))


def reconstruction_hook(activation, hook, sae_out, pos=None):
    # print("reconstruction l2 norm", (activation - sae_out).norm(dim=-1)[-3:, -3:])
    # print("reconstruction cos sim", cos_sim(activation, sae_out)[-3:, -3:])
    if pos is None:
        activation[:] = sae_out
    else:
        activation[:, pos] = sae_out[:, pos]
        
    return activation


def reconstruction_w_norm_correction_hook(activation, hook, sae_out, pos=None):
    activation_norm = activation.norm(dim=-1, keepdim=True)
    sae_out_norm = sae_out.norm(dim=-1, keepdim=True)
    corrected_activation = sae_out * (activation_norm / sae_out_norm)
    
    if pos is None:
        activation[:] = corrected_activation
    else:
        activation[:, pos] = corrected_activation[:, pos]
    
    return activation


def reconstruction_w_cos_correction_hook(activation, hook, sae_out, pos=None):
    activation_norm = activation.norm(dim=-1, keepdim=True)
    sae_out_norm = sae_out.norm(dim=-1, keepdim=True)
    corrected_activation = activation * (sae_out_norm / activation_norm)
    
    if pos is None:
        activation[:] = corrected_activation
    else:
        activation[:, pos] = corrected_activation[:, pos]
    
    return activation
    

def l2_error_preserving_perturbation_hook(activation, hook, sae_out, pos=None):
    error = (sae_out - activation).norm(dim=-1)
    perturbation = torch.randn_like(activation)
    normalized_perturbation = (
        perturbation / perturbation.norm(dim=-1, keepdim=True)
        ) * error.unsqueeze(-1)
    
    perturbed_activation = activation + normalized_perturbation
    
    # print("l2 perturbed l2 norm", (activation - perturbed_activation).norm(dim=-1)[-3:, -3:])
    # print("l2 perturbed cos sim", cos_sim(activation, perturbed_activation)[-3:, -3:])
    
    if pos is None:
        activation[:] = perturbed_activation
    else:
        activation[:, pos] = perturbed_activation[:, pos] 
    
    return activation


def cos_preserving_perturbation_hook(activation, hook, sae_out, preserve_sae_norm=False, pos=None):
    sae_out_norm = sae_out / sae_out.norm(dim=-1, keepdim=True)
    act_norm = activation / activation.norm(dim=-1, keepdim=True)
    
    reconstruction_cos_sim = einops.einsum(
        sae_out_norm, 
        act_norm, 
        "batch seq dim, batch seq dim -> batch seq"
    )    

    perturbation = torch.randn_like(act_norm)
    orthogonal_perturbation = perturbation - (act_norm * perturbation).sum(dim=-1, keepdim=True) * act_norm
    orthogonal_perturbation /= orthogonal_perturbation.norm(dim=-1, keepdim=True)

    perturbed_act = (
        reconstruction_cos_sim.unsqueeze(-1) * act_norm 
        + (1 - reconstruction_cos_sim.unsqueeze(-1) ** 2)**0.5 * orthogonal_perturbation
    )

    if preserve_sae_norm:
        perturbed_act *= sae_out.norm(dim=-1, keepdim=True)
    else:
        perturbed_act *= activation.norm(dim=-1, keepdim=True)
        
    if pos is None:
        activation[:] = perturbed_act
    else:
        activation[:, pos] = perturbed_act[:, pos] 
        
    return activation


def zero_ablation_hook(activation, hook, pos=None):
    zeros = torch.zeros_like(activation)
    if pos is None:
        activation[:] = zeros
    else:
        activation[:, pos] = zeros[:, pos]
    return activation


def mean_ablation_hook(activation, hook, pos=None):
    means = activation.mean((0, 1), keepdim=True).expand_as(activation)
    if pos is None:
        activation[:] = means
    else:
        activation[:, pos] = means[:, pos]
    return activation


def create_ablation_hooks(sae_out, pos=None, reshape_attn = False):
    ablation_hooks = [
        (
            'substitution', 
            partial(reconstruction_hook, sae_out=sae_out, pos=pos)),
        (
            'norm_corrected_substitution', 
            partial(reconstruction_w_norm_correction_hook, sae_out=sae_out, pos=pos)),
        (
            'cos_corrected_substitution', 
            partial(reconstruction_w_cos_correction_hook, sae_out=sae_out, pos=pos)),
        (
            'l2_error_preserving_substitution', 
            partial(l2_error_preserving_perturbation_hook, sae_out=sae_out, pos=pos)),
        (
            'cos_preserving_substitution_w_sae_norm', 
            partial(cos_preserving_perturbation_hook, sae_out=sae_out, pos=pos, preserve_sae_norm=True)),
        (
            'cos_preserving_substitution_w_true_norm', 
            partial(cos_preserving_perturbation_hook, sae_out=sae_out, pos=pos, preserve_sae_norm=False)),
        (
            'zero_ablation',
            partial(zero_ablation_hook, pos=pos)),
        (
            'mean_ablation', 
            partial(mean_ablation_hook, pos=pos))
    ]
    return ablation_hooks


def attn_hook_wrapper(activation, hook, hook_fn, n_heads=12, d_head=64):
    activation = einops.rearrange(
        activation,
        "batch seq n_heads d_head -> batch seq (n_heads d_head)")
    return einops.rearrange(
        hook_fn(activation, hook),
        "batch seq (n_heads d_head) -> batch seq n_heads d_head",
            n_heads=n_heads, d_head=d_head)
    

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


def run_error_eval_experiment(sae, model, token_tensor, layer, batch_size=64, pos=None, hook_loc="resid_pre"):
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
    parser.add_argument("--output_dir", type=str, default="error_eval_results")
    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    if args.hook_loc == "resid_pre":
        sae, model = load_sae(args.layer)
    elif args.hook_loc == "z":
        sae, model = load_attn_sae(args.layer)
    else:
        raise ValueError(f"Unsupported hook location {args.hook_loc}")
    
    sae = sae.to(args.device)
    model = model.to(args.device)
    
    token_tensor = torch.load("token_tensor.pt").to(args.device)
    
    result_df = run_error_eval_experiment(
        sae, 
        model, 
        token_tensor, 
        args.layer, 
        args.batch_size, 
        args.pos, 
        args.hook_loc
    )
    
    save_path = os.path.join(args.output_dir, f"gpt2_{args.hook_loc}")
    os.makedirs(save_path, exist_ok=True)
    pos_label = 'all' if args.pos is None else args.pos
    save_name = f"layer_{args.layer}_pos_{pos_label}.csv"
    
    result_df.to_csv(os.path.join(save_path, save_name), index=False)
