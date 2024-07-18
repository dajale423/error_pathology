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


## define hook functions
## write code in a way you have an activation, and you are moving towards another activation, whether it is random or real
def naive_random(activation, hook, length, pos=None):
    to_vector = torch.randn_like(activation)
    new_direction = (to_vector - activation)
    new_direction_unit_vector = new_direction/new_direction.norm(dim=-1, keepdim=True)

    perturbed_activation = activation + new_direction_unit_vector * length

    if pos is None:
        activation[:] = perturbed_activation
    else:
        activation[:, pos] = perturbed_activation[:, pos]    
    return activation

#use mean and covariance from activation
def covariance_random(activation, hook, multiNormal, length, device, pos=None):
    to_vector = multiNormal.sample().to(device)
    new_direction = (to_vector - activation)
    new_direction_unit_vector = new_direction/new_direction.norm(dim=-1, keepdim=True)

    perturbed_activation = activation + new_direction_unit_vector * length

    if pos is None:
        activation[:] = perturbed_activation
    else:
        activation[:, pos] = perturbed_activation[:, pos]    
    return activation

#point towards another random real direction
def real_direction(activation, hook, token_tensor, length, pos=None):
    tensor_length = token_tensor.shape[0]
    random_index = random.sample(range(tensor_length), 1)
    to_vector = token_tensor[random_index,:]
    
    new_direction = (to_vector - activation)
    new_direction_unit_vector = new_direction/new_direction.norm(dim=-1, keepdim=True)

    perturbed_activation = activation + new_direction_unit_vector * length

    if pos is None:
        activation[:] = perturbed_activation
    else:
        activation[:, pos] = perturbed_activation[:, pos]    
    return activation


def create_ablation_hooks(direction_type, device, pos= None, multiNormal = None, token_tensor = None):
    ablation_hooks = []
    length_list = list(range(1, 211, 10))
    if direction_type == "naive_random":
        for length in length_list:
            ablation_hooks.append((f'length_{length}', 
                                   partial(naive_random, length = length, pos=pos)))
    elif direction_type == "cov_random":
        for length in length_list:
            ablation_hooks.append((f'length_{length}', 
                                   partial(covariance_random, multiNormal= multiNormal, length = length,
                                           device=device, pos=pos)))
    elif direction_type == "real_direction":
        for length in length_list:
            ablation_hooks.append((f'length_{length}', 
                                   partial(real_direction, token_tensor=token_tensor,
                                           length = length, pos=pos)))

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

def run_error_eval_experiment(sae, model, token_tensor, layer, direction_type, device, seed,  batch_size=64, pos=None, hook_loc="resid_pre", e2e = None):
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    dataloader = torch.utils.data.DataLoader(
        token_tensor,
        batch_size=batch_size,
        shuffle=False
    )
    
    activation_loc = utils.get_act_name(hook_loc, layer)

    # generate normal distribution with mean and covariance from the known activations
    if direction_type != "naive_random":
        # get all activations
        first = True
        for ix, batch_tokens in enumerate(tqdm.tqdm(dataloader)):
            with torch.inference_mode():
                _, cache = model.run_with_cache(
                    batch_tokens, 
                    prepend_bos=True,
                    names_filter=[activation_loc]
                )
                activations = cache[activation_loc]
        
                if first:
                    all_activations = activations
                    first = False
                else:
                    all_activations = torch.cat((all_activations, activations))
        
        all_activations = einops.rearrange(all_activations, "batch seq n_hidden -> (batch seq) n_hidden")
        covariance = torch.cov(all_activations.T)
        mean = torch.mean(all_activations.T, dim = 1)

        if direction_type == "cov_random":
            multiNormal = torch.distributions.multivariate_normal.MultivariateNormal(mean.to("cpu"),
                                                                                     covariance.to("cpu"))
    
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

            if direction_type == "naive_random":
                ablation_hooks = create_ablation_hooks(direction_type=direction_type, device=device, pos=pos)
            elif direction_type == "cov_random":
                ablation_hooks = create_ablation_hooks(direction_type=direction_type, device=device, pos=pos,
                                                       multiNormal=multiNormal)
            elif direction_type == "real_direction":
                ablation_hooks = create_ablation_hooks(direction_type=direction_type, device=device, pos=pos,
                                                       token_tensor=all_activations)
            
            if hook_loc == "z":
                ablation_hooks = [
                    (name, partial(attn_hook_wrapper, hook_fn=hook_fn))
                    for name, hook_fn in ablation_hooks
                ]
            
            batch_result_df = run_all_ablations(model, batch_tokens, ablation_hooks, layer=layer, hook_loc=hook_loc)
            
            l0 = (feature_acts > 0).float().sum(dim=-1).cpu().numpy()[:, :-1].flatten()
            l1 = feature_acts.abs().sum(dim=-1).cpu().numpy()[:, :-1].flatten()
            reconstruction_error = (activations - sae_out).norm(dim=-1).cpu().numpy()[:, :-1].flatten()
            
            batch_result_df['norm'] = activations.norm(dim=-1).cpu().numpy()[:, :-1].flatten()
            batch_result_df['cos'] = cos_sim(activations, sae_out).cpu().numpy()[:, :-1].flatten()
            
            result_dfs.append(batch_result_df)
            
    return pd.concat(result_dfs).reset_index(drop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # layer, batchsize, num_batches, output_dir, pos 
    parser.add_argument("--hook_loc", type=str, default="resid_pre")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="sensitive_direction")
    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--e2e", type=str, default=None)
    parser.add_argument("--direction_type", type=str, default="naive_random",
                       choices = ["naive_random", "cov_random", "real_direction"])
    parser.add_argument("--seed", type=int, default=10)
    
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    
    print("loading sae and model")
    ## load a gpt2-small model
    args.e2e = "h9hrelni"

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
        args.direction_type,
        args.device,
        args.seed
        args.batch_size, 
        args.pos, 
        args.hook_loc,
        args.e2e
    )
    
    save_path = os.path.join("../results/" + args.output_dir, f"gpt2_{args.hook_loc}")
    os.makedirs(save_path, exist_ok=True)
    pos_label = 'all' if args.pos is None else args.pos
    
    save_name = f"{args.direction_type}_seed_{seed}_layer_{args.layer}_pos_{pos_label}.csv"

    # if args.e2e is not None:
    #     save_name = f"e2e_{args.e2e}_" + save_name
    
    result_df.to_csv(os.path.join(save_path, save_name), index=False)
