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


## define hook functions
## write code in a way you have an activation, and you are moving towards another activation, whether it is random or real
def naive_random(activation, hook, to_vector, length, pos=None):
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
def real_direction(activation, hook, all_activations, length, pos=None):
    tensor_length = all_activations.shape[0]
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


def create_ablation_hooks(direction_type, device, activations_shape, pos= None, multiNormal = None, all_activations = None, length_ranges="Normal"):
    ablation_hooks = []

    if length_ranges == "Normal":
        length_list = list(range(1, 20, 2)) + list(range(20, 50, 4)) + list(range(51, 211, 10))
    elif length_ranges == "Short":
        length_list = list(range(1, 30))
        length_list = [x/10 for x in length_list]
    
    if direction_type == "naive_random":
        to_vector = torch.randn(activations_shape)        
        for length in length_list:
            ablation_hooks.append((f'length_{length}', 
                                   partial(naive_random, to_vector = to_vector, length = length, pos=pos)))
    elif direction_type == "cov_random":
        for length in length_list:
            ablation_hooks.append((f'length_{length}', 
                                   partial(covariance_random, multiNormal= multiNormal, length = length,
                                           device=device, pos=pos)))
    elif direction_type == "real_direction":
        for length in length_list:
            ablation_hooks.append((f'length_{length}', 
                                   partial(real_direction, all_activations=all_activations,
                                           length = length, pos=pos)))

    return ablation_hooks

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

def get_all_activations(dataloader, model, activation_loc, e2e, remove_first_token = True):
    ## we want to save a tensor of active activations
    ## value of 1 for alive features, 0 for dead feature
    first = True
    with torch.inference_mode():
        for ix, batch_tokens in enumerate(tqdm.tqdm(dataloader)):
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

        if remove_first_token: # for skipping first token
            all_activations = all_activations[:, 1:, :]
    
        all_activations = einops.rearrange(all_activations, "batch seq n_hidden -> (batch seq) n_hidden")
    
        return all_activations


def run_error_eval_experiment(model, token_tensor, layer, direction_type, device,  batch_size=64, pos=None, 
                              hook_loc="resid_pre", e2e = None, remove_first_token = True, length_ranges = "Normal"):
    # sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    dataloader = torch.utils.data.DataLoader(
        token_tensor,
        batch_size=batch_size,
        shuffle=False
    )
    
    activation_loc = utils.get_act_name(hook_loc, layer)

    if direction_type != "naive_random":
        all_activations = get_all_activations(dataloader, model, activation_loc, e2e, remove_first_token)
        print(all_activations.shape)

        if direction_type == "cov_random":
            covariance = torch.cov(all_activations.T)
            torch.save(covariance, 'covariance.pt')
            mean = torch.mean(all_activations.T, dim = 1)

            # add a small value to ensure positive definite
            cov = covariance.clone()
            mask = cov.diagonal()
            cov += torch.diag(mask).bool().float() * 0.01
            
            multiNormal = torch.distributions.multivariate_normal.MultivariateNormal(mean.to("cpu"), cov.to("cpu"))
            del all_activations
        if direction_type == "real_direction":
            tensor_length = all_activations.shape[0]
            random_indexes = torch.randperm(tensor_length)[:500000]
            sampled_activations = all_activations[random_indexes,:]            
    
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
            

            if direction_type == "naive_random":
                ablation_hooks = create_ablation_hooks(direction_type=direction_type,  activations_shape = activations.shape, 
                                                   device=device, pos=pos, length_ranges=length_ranges)
            elif direction_type == "cov_random":
                ablation_hooks = create_ablation_hooks(direction_type=direction_type, device=device, pos=pos,
                                                       multiNormal=multiNormal, length_ranges=length_ranges)
            elif direction_type == "real_direction":
                ablation_hooks = create_ablation_hooks(direction_type=direction_type, device=device, pos=pos,
                                                       all_activations=all_activations, length_ranges=length_ranges)
            
            if hook_loc == "z":
                ablation_hooks = [
                    (name, partial(attn_hook_wrapper, hook_fn=hook_fn))
                    for name, hook_fn in ablation_hooks
                ]
            
            batch_result_df = run_all_ablations(model, batch_tokens, ablation_hooks, layer=layer, hook_loc=hook_loc, device=device)
            
            # l0 = (feature_acts > 0).float().sum(dim=-1).cpu().numpy()[:, :-1].flatten()
            # l1 = feature_acts.abs().sum(dim=-1).cpu().numpy()[:, :-1].flatten()
            # reconstruction_error = (activations - sae_out).norm(dim=-1).cpu().numpy()[:, :-1].flatten()
            
            batch_result_df['norm'] = activations.norm(dim=-1).cpu().numpy()[:, :-1].flatten()
            # batch_result_df['cos'] = cos_sim(activations, sae_out).cpu().numpy()[:, :-1].flatten()
            
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
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--length_ranges", type=str, default="Normal")
    
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"running {args.direction_type}")

    
    print("loading sae and model")
    ## load gpt2-small model, but choose any sae model (since the sae model doesn't matter)
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
        # sae = sae_transformer_model.saes[f"blocks-{args.layer}-hook_{args.hook_loc}"]

    # sae = sae.to(args.device)
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
        model, 
        token_tensor, 
        args.layer,
        args.direction_type,
        args.device,
        args.batch_size, 
        args.pos, 
        args.hook_loc,
        args.e2e,
        length_ranges = args.length_ranges
    )
    
    save_path = os.path.join("../results/" + args.output_dir, f"gpt2_{args.hook_loc}")
    os.makedirs(save_path, exist_ok=True)
    pos_label = 'all' if args.pos is None else args.pos

    save_name = f"{args.direction_type}_seed_{args.seed}_layer_{args.layer}_pos_{pos_label}.csv"
    
    if args.length_ranges == "Short":
        save_name = f"{args.direction_type}_seed_{args.seed}_layer_{args.layer}_pos_{pos_label}_short.csv"
    
    result_df.to_csv(os.path.join(save_path, save_name), index=False)
