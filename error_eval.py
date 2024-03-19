from functools import partial

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformer_lens import utils


def reconstruction_hook(activation, hook, sae_out):
    activation[:] = sae_out
    return activation


def reconstruction_w_norm_correction_hook(activation, hook, sae_out):
    activation_norm = activation.norm(dim=-1, keepdim=True)
    sae_out_norm = sae_out.norm(dim=-1, keepdim=True)
    activation[:] = sae_out * (activation_norm / sae_out_norm)
    return activation


def reconstruction_w_cos_correction_hook(activation, hook, sae_out):
    activation_norm = activation.norm(dim=-1, keepdim=True)
    sae_out_norm = sae_out.norm(dim=-1, keepdim=True)
    activation[:] = activation * (sae_out_norm / activation_norm)
    return activation
    

def l2_error_preserving_perturbation_hook(activation, hook, sae_out):
    error = (sae_out - activation).norm(dim=-1)
    perturbation = torch.randn_like(activation)
    normalized_perturbation = (
        perturbation / perturbation.norm(dim=-1, keepdim=True)
        ) * error.unsqueeze(-1)
    
    return activation + normalized_perturbation


def cos_preserving_perturbation_hook(activation, hook, sae_out, preserve_sae_norm=False):
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
        
    return perturbed_act


def zero_ablation_hook(activation, hook):
    return torch.zeros_like(activation)


def mean_ablation_hook(activation, hook):
    return activation.mean((0, 1), keepdim=True).expand_as(activation)


def run_all_ablations(model, batch_tokens, sae_out, layer, hook_loc="resid_pre"):
    hooks = [
        ('substitution', partial(reconstruction_hook, sae_out=sae_out)),
        ('norm_corrected_substitution', partial(reconstruction_w_norm_correction_hook, sae_out=sae_out)),
        ('cos_corrected_substitution', partial(reconstruction_w_cos_correction_hook, sae_out=sae_out)),
        ('l2_error_preserving_substitution', partial(l2_error_preserving_perturbation_hook, sae_out=sae_out)),
        ('cos_preserving_substitution_w_sae_norm', partial(cos_preserving_perturbation_hook, sae_out=sae_out, preserve_sae_norm=True)),
        ('cos_preserving_substitution_w_true_norm', partial(cos_preserving_perturbation_hook, sae_out=sae_out, preserve_sae_norm=False)),
        ('zero_ablation', zero_ablation_hook),
        ('mean_ablation', mean_ablation_hook)
    ]
    
    orginal_logits = model.run_with_hooks(batch_tokens)
    
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
    
    for hook_name, hook in hooks:
        
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