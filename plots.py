import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F


def format_subplot(ax, grid_x=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_x:
        ax.grid(linestyle='--', alpha=0.4)
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4)


def plot_layerwise_kl_average(result_dfs, layers, kl_cols, kl_col_labels):
    mean_df = pd.DataFrame({
        l: result_dfs[l].mean(axis=0) for l in layers
    }).T

    fig, ax = plt.subplots(figsize=(10, 5))

    mean_df[kl_cols].rename(columns=kl_col_labels).plot(kind='bar', ax=ax)

    ax.set_ylabel('KL(original prediction || prediction w/ substitution)', fontsize=11)
    ax.set_xlabel('substitution layer', fontsize=12)
    ax.legend(title='Substitution type', fontsize=12, title_fontsize=14)
    
    format_subplot(ax)
    
    return fig, ax


def plot_layerwise_average_loss_increase(result_dfs, layers, loss_cols, loss_col_labels):
    mean_df = pd.DataFrame({
        l: result_dfs[l].mean(axis=0) for l in layers
    }).T

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_cols = []
    for col in loss_cols:
        increase_col_name = col + '_increase'
        mean_df[increase_col_name] = mean_df[col] - mean_df['loss']
        loss_col_labels[increase_col_name] = loss_col_labels[col]
        plot_cols.append(increase_col_name)
        
    mean_df[plot_cols].rename(columns=loss_col_labels).plot(kind='bar', ax=ax)

    ax.set_ylabel('absolute cross-entropy loss change', fontsize=12)
    ax.set_xlabel('substitution layer', fontsize=12)
    ax.legend(title='Substitution type', fontsize=12, title_fontsize=14)
    
    format_subplot(ax)
    
    return fig, ax, mean_df


def plot_top_token_kl_comparison(nominal_logits, sae_logits, random_logits, seq, pos, k=25, ax=None):
    
    original_log_probs = nominal_logits.log_softmax(dim=-1).cpu()
    sae_log_probs = sae_logits.log_softmax(dim=-1).cpu()
    random_log_probs = random_logits.log_softmax(dim=-1).cpu()
    
    sae_patch_kl_div = F.kl_div(
        sae_log_probs, 
        original_log_probs,
        log_target=True, 
        reduction='none'
    ).sum(dim=-1)

    random_patch_kl_div = F.kl_div(
        random_log_probs, 
        original_log_probs,
        log_target=True, 
        reduction='none'
    ).sum(dim=-1)

    e = 2.718
    sae_kl = sae_patch_kl_div[seq, pos].item()
    random_kl = random_patch_kl_div[seq, pos].item()
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        
    orginal_top_log_p_value, orignal_top_indices = original_log_probs[seq, pos].sort(descending=True)
    ax.plot(orginal_top_log_p_value[:k].cpu().numpy(), label='original')
    ax.plot(sae_log_probs[seq, pos][orignal_top_indices[:k]].cpu().numpy(), label=f'SAE (KL={sae_kl:.3f})')
    ax.plot(random_log_probs[seq, pos][orignal_top_indices[:k]].cpu().numpy(), label=f'$\epsilon$-random (KL={random_kl:.3f})')

    ax.legend(loc='lower left', frameon=False)

    top_log_prob = orginal_top_log_p_value[0]
    top_sae_log_prob = sae_log_probs[seq, pos][orignal_top_indices[0]]
    top_rand_log_prob = random_log_probs[seq, pos][orignal_top_indices[0]]

    top_prob = e ** top_log_prob
    sae_top_prob = e ** top_sae_log_prob
    random_top_prob = e ** top_rand_log_prob

    # add horizontal line from x = 0 to x=5 at y = -1.5
    ax.axhline(y=top_log_prob, color='tab:blue', linestyle='--', xmin=0, xmax=0.2, lw=0.75)
    ax.axhline(y=top_sae_log_prob, color='tab:orange', linestyle='--', xmin=0, xmax=0.37, lw=0.75)
    ax.axhline(y=top_rand_log_prob, color='tab:green', linestyle='--', xmin=0, xmax=0.54, lw=0.75)

    # add annotation to right of the horizontal line with the prob value
    ax.annotate(f'p={top_prob:.3f}', (3.35, top_log_prob), textcoords="offset points", xytext=(10,0), ha='left', va='center', fontsize=9, color='tab:blue')
    ax.annotate(f'p={sae_top_prob:.3f}', (7.5, top_sae_log_prob), textcoords="offset points", xytext=(10,0), ha='left', va='center', fontsize=9, color='tab:orange')
    ax.annotate(f'p={random_top_prob:.3f}', (12, top_rand_log_prob), textcoords="offset points", xytext=(10,0), ha='left', va='center', fontsize=9, color='tab:green')

    ax.set_xlabel('token rank')