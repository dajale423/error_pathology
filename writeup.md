# SAE reconstruction errors are pathological

## Summary
Sparse Autoencoder (SAE) errors are pathological -- substituting an SAE reconstructed vector $x_{SAE}$ for the original activation vector $x$ with error $\|x_{SAE} - x\|_2 = \epsilon$ changes the next token probabilities (i.e., increases KL) significantly and consistently more than a random vector $x_{\epsilon}$ where $\|x_{\epsilon} - x\|_2 = \epsilon$. This is true for all layers of the model (2.2x to 4.5x increase in KL over baseline), is cross-entropy loss increasing, and the cause is not [feature suppression](https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes).

TODO: insert main layer plot

### Intuition: how big a deal is this (KL) difference?
For some intuition, here are several real examples of what the top-25 token probabilities look like when patching in SAE and $\epsilon$-random reconstructions compared to the original model next-token distribution (note the log scale). 

For additional intuition on KL divergence, see [this excellent post](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence).


## Introduction
As the interpretability community allocates more resources and increases reliance on SAEs, it is important to understand the limitation and potential flaws of this method.

SAEs are designed to find a sparse overcomplete-feature basis for a model's latent space. This is done by minimizing the reconstruction error of the input data with a sparsity penalty: 
$$\min_{SAE} \|x - SAE(x) \|_2^2 + \lambda \|SAE(x)\|_1.$$ 
However, the true goal is to find a faithful feature decomposition (i.e., a decomposition which accurately captures the true causal variables in the model) and reconstruction error is only an easy-to-optimize proxy objective. This begs the questions: how good of a proxy objective is this? How faithful are the reconstructed representations? Are we potentially proxy gaming?

A natural property of a "faithful" reconstruction is that substituting the reconstruction should approximately preserve the next-token prediction probabilities. More formally, for a set of tokens $T$ and a model $M$, let $P = M(T)$ be the models true next token probabilities. Then let $Q_{SAE} = M(T | do(x <- SAE(x)))$ be the next token probabilities after intervening on the model by replacing a particular activation $x$ (e.g. a residual stream state or a layer of MLP activations) with the SAE reconstruction of $x$. The more faithful the reconstruction, the lower the KL divergence between $P$ and $Q$ (denoted as $D_{KL}(P || Q_{SAE})$) should be.

In this post, I study how $D_{KL}(P || Q_{SAE})$ compares to several natural baselines based on random perturbations of activations vectors which preserve some error property of the SAE construction (e.g., having the same $l_2$ reconstruction error or cosine similarity). I find that the KL divergence is significantly higher (2.2x - 4.5x) for the residual stream SAE reconstruction than for the random perturbations, suggesting that the SAE reconstruction is not faithful in the sense that it does not preserve the next token probabilities.

This observation is important because it suggests that SAEs are making systematic errors which miss an important part of the true representational structure of the model. Moreover, as we improve our SAE training techniques, its possible that we will exploit this proxy gap, rather than improving the true faithfulness of the reconstruction. The good news is that this gap presents a clear target for methodological improvement and a new metric for evaluating SAEs.

## Experiments and Results
I conduct experiments on Joseph Bloom's GPT2-small residual stream [SAEs](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream) with 32x expansion factor on 2 million tokens (16k sequences of length 128).

My code can be found in [this](https://github.com/wesg52/mats_sae_training/tree/error_pathology) fork branch.

### Intervention Types
To evaluate the faithfulness of the SAE reconstruction, I consider several types of interventions. Assume that $x$ is the original activation vector and $x_{SAE}$ is the SAE reconstruction of $x$.

- $\epsilon$-random substitution: $x_{\epsilon}$ is a random vector with $\|x_{\epsilon} - x\|_2 = \|x_{SAE} - x\| = \epsilon$. I.e., both $x_{SAE}$ and $x_\epsilon$ are random vectors on the $\epsilon$-ball around $x$.
- $\theta$-random substitution: $x_{\theta}$ is a random vector with $\cos(x_{\theta}, x) = \cos(x_{SAE}, x) = \cos(\theta)$. I consider both versions where the norm of $x_{\theta}$ is adjusted to be $\|x\|$ and $\|x_{SAE}\|$. 
- SAE-norm substitution: this is the same as the original activation vector except the norm is altered to the SAE norm $x^\prime = (\|x_{SAE}\| / \|x\|) * x$. This is a baseline to isolate the effect of the norm change from the SAE reconstruction, a known pathology identified [here](https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes).
- norm-corrected SAE substitution: this is the same as $x_{SAE}$  except the norm is altered to the true norm $x_{SAE}^\prime = (\|x\| / \|x_{SAE}\|) * x_{SAE}$. Similar motivation as above.

In addition to these different kinds of perturbations, I also consider applying the perturbation to all tokens in the context and just applying it to a single token. This is to test the hypothesis that the pathology is caused by compounding and correlated errors (since the $\epsilon$-random substitution errors are uncorrelated).

Here is are the average KL differences (across 2M tokens) for each intervention when intervened across all tokens in the context:

TODO: insert all intervention plot

There are 3 clusters of error severity

- The $x_{SAE}$ and norm corrected $x_{SAE}$ are both high with norm corrected slightly higher (this makes sense because this increases the l2 reconstruction error). 
- $\epsilon$-random and both variants of $\theta$-random have much lower but non-trivial KL compared to the SAE reconstruction. They are all about the same because randoms vector in a high dimensional space are almost surely almost orthogonal so the $\epsilon$-random perturbation has an effect similar to the $\theta$-random perturbation.
- Most importantly, the SAE-norm substitution has an almost 0 KL divergence. This is important because it shows that the difference is not caused by the smaller norm (a known problem with SAEs) but the direction.

For this reason, in the rest of the post we mostly focus on the $\epsilon$-random substitution as the most natural baseline.


### Layerwise Intervention Results in More Detail
Next, I consider distributional statistics to get a better sense for how the errors are distributed and how this varies between layers.

This is a histogram of the KL differences for all layers under $\epsilon$-random substitution and the SAE reconstruction (and since I clip the tails at 1.0 for legibility, I also report the 99.9th percentile). Again the substitution happens for all tokens in the context but the substitution is for a single layer. Note the log scale.

TODO: insert layer KL hist


Here is the same plot but instead of KL divergence, I plot the cross-entropy loss difference (with mean instead of 99.9p). While KL measures deviation from the original distribution, the loss difference measures the *degradation* in the model's ability to predict the true next token.

TODO: insert loss hist

Just as with KL, the mean loss increase of the SAE substitution is 2-4x higher compared to the $\epsilon$-random baseline.

- All layer KL hist
- All layer loss hist
- All layer KL by position

### Single Token Intervention Results

Mostly just a bunch of plots.

- Plot of all layer KL differences histogram under all token ablation
- Plot all layer KL differences under single token ablation
- Plot all layer loss differences
- Discuss different kinds of ablations (show this is not the same thing as the norm difference pathology)
- For same token, sample many different random perturbations to get confidence
- Plot ablation means by position

- Examples of where the KL difference is highest

### Replication with Attention SAEs

### How pathological are the errors?

### When do these errors happen?

## Concluding Thoughts
### Why is this happening?
- Some features are dense (eg, position is 20-dense)

### Takeaways

- *Both* SAE and $\epsilon$-random substitution KL divergence should be a standard SAE evaluation metric to measure faithfulness of the reconstruction.
- Generally, loss recovered seems a worse metric than KL divergence. E.g., consider the case where both the original model and the SAE substituted model have place probability $p$ on the correct token but their top token probabilities are all different. Loss recovered will imply that the reconstruction is perfect when it is actually quite bad.
- Closing the gap between SAE and $\epsilon$-random substitution KL divergence is a promising direction for future work.

Future work:

- Compounding SAE substitution errors -- I bet multiple SAE substitution errors compound more than multiple $\epsilon$-random substitution errors.
- Is there a point in training where there is a divergence?
- Better understand the cases where the KL divergence is highest and what causes this pathology in general?