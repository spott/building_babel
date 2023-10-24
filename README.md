# Building Babel

This is an attempt at a progressively growing GPT-style language model.

The idea is to start small (55M parameters or so), and iteratively train the model while increasing both vocabulary size
and model size until we reach something approaching some of the larger language models (1-7B is probably the limit that I will be able to afford)

## Motivations:

Training a modern LLM is an expensive endeavor, requiring significant compute resources and dataset sizes.  Even a "small" 1 Billion parameter can 
take up to \$5k to train, and training costs for Lllama2 7B are likely over \$100k at curent A100 reserved prices.  Training a model progressively
has a lot of value in reducing this cost drastically, as long as you can maintain performance.

The way we currently train language models is an "all at once" approach.  We continually throw data from a huge vocabulary, with a hugely varying sentence structure
and just wait till they learn enough to be coherent as they respond.  We are asking models to learn everything all at once, at the same time, rather than learn through some curriculum
similar to how humans learn.

This "learn everything at the same time" method also likely leads to a less structured language understanding within the model than a progressive learning can 
(theoretically) accomplish, as we can reduce learning rates on parts of the model that have already been learned in order to allow for structure to be maintained.

## Method:

We will be building a modern decoder only transformer based language model similar to llama 2 in order to give us a good baseline for comparison.  Thus we will be using

* Pre-normalization, assuming this doesn't screw with our ability to expand our model representation dimension.
* Swiglu activation (and 8d/3 feedforward network size)
* RoPE positional embeddings.

We will also be reusing the llama2 tokenizer.  Note that the tokenizer will be fixed throughout training, while the embedding space will expand.  We will also keep the individual head dimensions fixed at 128 (same as llama).

We will be progressively growing this model from approximately 55M parameters, and doubling the parameter size as we increase the vocabulary size.  We will increase the parameter space by increasing *either* the embedding dimension (and thus the number of heads of attention in each layer, while fixing the dimension of each head), or the number of layers.  We will use zero-init attention for each new attention layer and head, to allow the model to continue to use it's current understanding even after a model size change.

### Implementation questions:

* The optimizer keeps a record of every parameter in the model in order to continue training it... do we need to preserve the optimizers record?
* if we want to have different learning rates for different parts of the model, I think we *have* to have the optimizer keep track of different 
* do we zero-init the embedding space as we expand it? or randomly initialize it?
* How do we deal with the learning rate on previously learned parts of the model?
* How do we deal with the residual connection after each attention head (especially if we have separate learning rates for the previous part of the model)?
* How do we deal with increasing the size of the feed forward network?  (maybe use the gating mechanism?)

## Further work:

For simplicity sake, we only changed the vocabulary size, without changing the sentence structure difficulty or specifically clustering domain specific information.
