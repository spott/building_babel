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

* The optimizer (adam) keeps a record of every parameter in the model in order to continue training it... do we need to preserve the optimizers record?
* do we zero-init the embedding space as we expand it? or randomly initialize it?
    * This is going to have a few different options.  for new layers, we zero-init things, for ffns, we zero-init the gate, and w2, but w1 we kaiming_init, for attention, we zero-init new parts of the transfomer matrix, except the 'lr' part, which is an eye.  In general we want the model to continue to work after growing (producing the same results), but init can help when we can.
* How do we deal with the learning rate on previously learned parts of the model?
    * we will keep this rate decaying in at the same rate it was.
* How do we deal with the residual connection after each attention head (especially if we have separate learning rates for the previous part of the model)?
* How do we deal with increasing the size of the feed forward network?  (maybe use the gating mechanism?)
* We *probably* don't want parallel layers (where FFN and Attn are done in parallel rather than one after the other), as this 
doesn't do well for smaller models, and we will be small most of the time.

### Todo:
* GrowableLinear needs to be parallized.
* GrowableLinear might need a way to condense generations in to get some speedups (so maybe after 2 gens, we fuse the learning rates of the all smaller gens).
* GrowableLinear might need to figure out how to not create an empty parameter set when one dim doesn't grow (like vocab_size)
* GrowableEmbedding needs to be created...
* Flops counter needs to be created.
* parameter counter needs to be created.
* RoPE needs to be done.  Using [this](https://github.com/foundation-model-stack/foundation-model-stack/blob/main/fms/modules/positions.py) could be a good place to start.
* we might need to have a way to grow the individual head_dim... otherwise we might not be able to have the same head_dim for small and large models.

## Further work:

For simplicity sake, we only changed the vocabulary size, without changing the sentence structure difficulty or specifically clustering domain specific information.
