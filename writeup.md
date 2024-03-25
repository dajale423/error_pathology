# SAE reconstruction errors are pathological

## Summary
Sparse Autoencoder (SAE) errors are pathological -- substituting an SAE reconstruction vector $x_{SAE}$ for the orginal activation vector $x$ with error $\|x_{SAE} - x\|_2 = \epsilon$ changes the next token probabilities (i.e., increases KL) significantly and consistently more than a random vector $x_{\epsilon}$ where $\|x_{\epsilon} - x\|_2 = \epsilon$. This is true for all layers of the model, is cross-entropy loss increasing, and the cause is not [feature suppression](https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes). This suggests that the inductive biases of SAEs are missing 

TODO: insert main layer plot

### Intuition: how big a deal is this KL difference?
For some intuition, here are several real examples of what the top-25 token probabilities look like when patching in SAE and $\epsilon$-random reconstructions compared to the original model next-token distribution. 


For additional intuition on KL divergence, see [this excellent post](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence).


## Introduction
As the interpretability community allocates more resources and increases reliance on SAEs, it is important to understand the limitation and potential flaws of this method.

SAEs are designed to find a sparse overcomplete-feature basis for a model's latent space. This is done by minimizing the reconstruction error of the input data with a sparsity penalty: 
$$\min_{SAE} \|x - SAE(x) \|_2^2 + \lambda \|SAE(x)\|_1.$$ 
However, given the goal is to find a faithful feature decomposition (i.e., a decomposition which accurately captures the true causal variables in the model), reconstruction error is only an easy-to-optimize for proxy objective. An important question is how good of a proxy objective is this?

A natural property of a "faithful" reconstruction is that substituting the reconstruction should approximately preserve the next-token probabilites. Hence  A less proxy objective is that the final next-token probabilities are the approximately the same. More formally, for a set of tokens $T$ and a model $M$, let $P = M(T)$ be the models true next token probabilites. Then let $Q = M(T | do(x <- SAE(x)))$ be the next token probabilities after intervening on the model by replacing a particular activation $x$ (e.g. a residual stream state or a layer of MLP activations) with the SAE reconstruction of $x$. Then a less proxy objective is the KL divergence between $P$ and $Q$ denoted as $D_{KL}(P || Q)$.

In this post, we study how

Important be


## Experiments and Results
- Conduct experiments on Joseph Bloom's GPT2-small residual stream SAEs with 32x expansion factor with 2 million tokens (16k sequences of length 128).

### Intervention Types
Replacement


Mostly just a bunch of plots.

- Plot of all layer KL differences histogram under all token ablation
- Plot all layer KL differences under single token ablation
- Plot all layer loss differences
- Discuss different kinds of ablations (show this is not the same thing as the norm difference pathology)
- For same token, sample many different random perturbations to get confidence
- Plot ablation means by position

- Examples of where the KL difference is highest

## Conclusion
### Why is this happening?
- Some features are dense (eg, position is 20-dense)