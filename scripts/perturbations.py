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
from error_eval import cos_sim, load_sae, load_attn_sae
import random 

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
from perturbations import run_all_ablations


def run_all_ablations(model, batch_tokens, ablation_hooks, layer, device, hook_loc):
    
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

    final_resid_post_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], 768), device=device)
    perturbed_final_resid_post_store = torch.zeros((batch_tokens.shape[0], batch_tokens.shape[1], 768), device=device)

    def get_activation(
        activation, hook
    ):
        '''
        Get the activation
        '''
        final_resid_post_store[:, :] = activation[:, :].detach()
    
    def get_activation_perturbed(
        activation, hook
    ):
        '''
        Get the activation
        '''
        perturbed_final_resid_post_store[:, :] = activation[:, :].detach()

    activation_loc_end = utils.get_act_name("resid_post", 11)
    model.run_with_hooks(
        batch_tokens, 
        return_type=None, # For efficiency, we don't need to calculate the logits
        fwd_hooks=[(activation_loc_end, get_activation)]
    )

    
    for hook_name, hook in ablation_hooks:
        
        # intervention_logits = model.run_with_hooks(
        #     batch_tokens,
        #     fwd_hooks=[(utils.get_act_name(hook_loc, layer), hook)]
        # )
        intervention_logits = model.run_with_hooks(
            batch_tokens,
            fwd_hooks=[(utils.get_act_name(hook_loc, layer), hook),
                      (activation_loc_end, get_activation_perturbed)]
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

        intervention_final_L2 = torch.linalg.vector_norm(final_resid_post_store - perturbed_final_resid_post_store, ord=2, dim=-1)
        
        batch_result_df[hook_name + "_loss"] = intervention_loss.flatten().cpu().numpy()
        batch_result_df[hook_name + "_kl"] = intervention_kl_div[:, :-1].flatten().cpu().numpy()
        batch_result_df[hook_name + "_blocks.11.hook_resid_post_L2"] = intervention_final_L2[:, :-1].flatten().cpu().numpy()
    
    return batch_result_df