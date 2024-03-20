from functools import partial

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformer_lens import utils


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
        
    return perturbed_act


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


def create_ablation_hooks(sae_out, pos=None):
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