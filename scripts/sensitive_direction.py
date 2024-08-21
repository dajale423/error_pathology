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



## define hook functions
## write code in a way you have an activation, and you are moving towards another activation, whether it is random or real
def towards_a_vector(activation, hook, to_vector, length, pos=None):
    new_direction = (to_vector - activation)
    new_direction_unit_vector = new_direction/new_direction.norm(dim=-1, keepdim=True)
    perturbed_activation = activation + new_direction_unit_vector * length

    if pos is None:
        activation[:] = perturbed_activation
    else:
        activation[:, pos] = perturbed_activation[:, pos]    
    return activation

def perturb_along_vector(activation, hook, perturb_vector, length, pos=None):
    new_direction_unit_vector = perturb_vector/perturb_vector.norm(dim=-1, keepdim=True)
    perturbed_activation = activation + new_direction_unit_vector * length

    if pos is None:
        activation[:] = perturbed_activation
    else:
        activation[:, pos] = perturbed_activation[:, pos]    
    return activation

# #use mean and covariance from activation
# def covariance_random(activation, hook, to_vector, length, pos=None):
#     print(to_vector.shape)
#     print(activation.shape)
#     new_direction = (to_vector - activation)
#     new_direction_unit_vector = new_direction/new_direction.norm(dim=-1, keepdim=True)
#     perturbed_activation = activation + new_direction_unit_vector * length

#     if pos is None:
#         activation[:] = perturbed_activation
#     else:
#         activation[:, pos] = perturbed_activation[:, pos]    
#     return activation

#point towards another random real direction
# def real_direction(activation, hook, all_activations, length, pos=None):
#     tensor_length = all_activations.shape[0]
#     random_index = random.sample(range(tensor_length), 1)
#     to_vector = token_tensor[random_index,:]
    
#     new_direction = (to_vector - activation)
#     new_direction_unit_vector = new_direction/new_direction.norm(dim=-1, keepdim=True)

#     perturbed_activation = activation + new_direction_unit_vector * length

#     if pos is None:
#         activation[:] = perturbed_activation
#     else:
#         activation[:, pos] = perturbed_activation[:, pos]    
#     return activation


def create_ablation_hooks(direction_type, subtraction, device, activations_shape, pos= None, multiNormal = None, all_activations = None, length_ranges="Normal"):
    ablation_hooks = []
    

    if length_ranges == "Normal":
        length_list = list(range(1, 20, 2)) + list(range(20, 50, 4)) + list(range(51, 101, 10))
    elif length_ranges == "Short":
        length_list = list(range(1, 30))
        length_list = [x/10 for x in length_list]
        
    if "naive_random" in direction_type:
        to_vector = torch.randn(activations_shape).to(device)
        to_vector = to_vector * 121 #121 is the median norm in layer 6
    elif "cov_random" in direction_type:
        if subtraction == "mixture":
            vector_1 = multiNormal.sample(sample_shape = [activations_shape[0], activations_shape[1]]).to(device)
            vector_2 = multiNormal.sample(sample_shape = [activations_shape[0], activations_shape[1]]).to(device)
            to_vector = vector_1 - vector_2
        elif subtraction == "itself":
            to_vector = multiNormal.sample(sample_shape = [activations_shape[0], activations_shape[1]]).to(device)
    elif direction_type == "real_direction":
        tensor_length = all_activations.shape[0]
        activation_num = activations_shape[0] * activations_shape[1] # get number of activations to sample
        random_indexes = torch.randperm(tensor_length)[:activation_num]
        to_vector = all_activations[random_indexes,:]
        to_vector = einops.rearrange(to_vector, "(batch seq) n_hidden -> batch seq n_hidden", batch = activations_shape[0])
    elif direction_type == "zero":
        to_vector = torch.zeros(activations_shape).to(device)


    if subtraction == "mixture": # perturb along the vector
        for length in length_list:
            ablation_hooks.append((f'length_{length}', 
                               partial(perturb_along_vector, perturb_vector = to_vector, length = length, pos=pos)))
    elif subtraction == "itself":
        for length in length_list:
            ablation_hooks.append((f'length_{length}', 
                                   partial(towards_a_vector, to_vector = to_vector, length = length, pos=pos)))
    return ablation_hooks


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


def run_error_eval_experiment(model, token_tensor, layer, direction_type, subtraction, device,  batch_size=64, pos=None, 
                              hook_loc="resid_pre", e2e = None, remove_first_token = True, length_ranges = "Normal"):
    # sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads

    dataloader = torch.utils.data.DataLoader(
        token_tensor,
        batch_size=batch_size,
        shuffle=False
    )
    
    activation_loc = utils.get_act_name(hook_loc, layer)

    if ("cov_random" in direction_type) |("real_direction" in direction_type):
        all_activations = get_all_activations(dataloader, model, activation_loc, e2e, remove_first_token)

        if "cov_random" in direction_type:
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
            if subtraction == "mixture":
                tensor_length = all_activations.shape[0]
                random_indexes = torch.randperm(tensor_length)
                all_subtractions = all_activations - all_activations[random_indexes]
                # remove all rows where subtracted itself
                all_subtractions = all_subtractions[all_subtractions.abs().sum(dim=1) != 0]
                del all_activations
    
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
            

            if (direction_type == "naive_random") or (direction_type == "zero"):
                ablation_hooks = create_ablation_hooks(direction_type=direction_type, subtraction=subtraction,  
                                                       activations_shape = activations.shape, 
                                                   device=device, pos=pos, length_ranges=length_ranges)
            elif "cov_random" in direction_type:
                ablation_hooks = create_ablation_hooks(direction_type=direction_type, subtraction=subtraction,
                                                       activations_shape = activations.shape, device=device, pos=pos,
                                                       multiNormal=multiNormal, length_ranges=length_ranges)
            elif direction_type == "real_direction":
                if subtraction == "itself":
                    ablation_hooks = create_ablation_hooks(direction_type=direction_type, subtraction=subtraction, 
                                                           activations_shape = activations.shape, device=device, pos=pos,
                                                           all_activations=all_activations, length_ranges=length_ranges)
                elif subtraction == "mixture":
                    ablation_hooks = create_ablation_hooks(direction_type=direction_type, subtraction=subtraction, 
                                                           activations_shape = activations.shape, device=device, pos=pos,
                                                           all_activations=all_subtractions, length_ranges=length_ranges)
            
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
                       choices = ["zero", "naive_random", "cov_random", "real_direction"])
    parser.add_argument("--subtraction", type=str, default="itself",
                       choices = ["itself", "mixture"])
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--length_ranges", type=str, default="Normal",
                       choices = ["Normal", "Short"])
    
    args = parser.parse_args()

    # set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"running {args.direction_type}, {args.subtraction}")
    print(f"layer {args.layer}")
    print(f"length ranges:  {args.length_ranges}")
    
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
        args.subtraction,
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

    save_name = f"{args.direction_type}_{args.subtraction}_seed_{args.seed}_layer_{args.layer}_pos_{pos_label}"
    
    if args.length_ranges == "Short":
        save_name = save_name + "_short"
    result_df.to_csv(os.path.join(save_path, save_name + ".csv"), index=False)
