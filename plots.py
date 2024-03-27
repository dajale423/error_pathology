import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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