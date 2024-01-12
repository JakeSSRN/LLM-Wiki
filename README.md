# General Terminology

**Inference** - What LLMs are designed to do - infer what words come next in a sequence. Can be done via CPU, GPU or a combination of both. GPU inference is roughly 10x faster than CPU inference, and any use of the CPU will greatly slow down inference speed. VRAM in consumer GPUs goes up to 24 GB currently. Many models are bigger than this, so CPU inference is required to run larger models on consumer hardware.

Note: if you have deep pockets, you can run multiple GPUs in parallel with minimal sacrifice to inference speed. Alternatively, you can pick up enterprise grade hardware. The current king of this is the NVIDIA H100 card with 80 GB of VRAM. They go for about $30k each. Good luck getting your hands on one.

**Model** - The neural network architecture designed to process and generate text. They are trained on a narrow or wide variety of subjects and tasks. It is best to select a model trained for what you would like to do with it. Have a collection of models for specific tasks, rather than trying to find an excellent all-around model, especially if you cannot easily run models larger than 13B.

**Parameters (B number)**- pieces of knowledge within the model’s training, usually in the billions range. The greater the number of parameters, the more the model “knows”, and so the better it will be at applying that knowledge in comprehensive, connective ways. Models with more knowledge tend to create “smarter” responses. More parameters also takes up more memory.

**Tokens** - the smallest chunk of text a model can process. Like syllables in language. A good rule of thumb is that for English, a token is 0.75 words. 4096 tokens, a common context length, will be about 3,072 words. 32k tokens will be about 24k words.

**Weights** - numerical values assigned to the edges (connections) and between neurons (nodes) within the neural network during training. These represent the strength of the connections, and are adjusted iteratively during training. The weights determine the influence of each connection on the final output and play a critical role in shaping the model’s behavior and performance.

**Perplexity** - a score measuring uncertainty when the model is inferencing. Lower indicates better quality - generally, more intelligent and coherent responses from the model.

**Compute** - a term that in this context has come to refer to raw computing power. If something needs a lot of compute, say training a base model, it requires significant processing power in high density. With the price of enterprise grade hardware and its current necessity in developing AI, there is some concern regarding large corporation’s comparative “monopoly” on compute.

**Context window** - the amount of tokens the model can keep in its “short term memory”. Once the interaction has exceeded this number of tokens, the model will no longer be aware of their content. The model will also be unable to generate a coherent response longer than its context length. Context window is generally established in training, though it can be altered somewhat through other methods discussed further down.

**SOTA (State of the Art)** - A tag indicating a novel concept or application.

# Quantization
Most models are developed in 32-bit floating point (FP32) representation. This is extremely precise and captures nuanced data well, but also takes up a huge amount of space. A general rule of thumb is that 32 bit models require 4 GB of memory (RAM or VRAM) per 1 billion parameters, so a 7B model would require 28 GB or memory. This is not attainable on most consumer hardware, so models are quantized to reduce resource demand.

Quantization is basically compression, folding lots of individual pieces of information into a single piece of information that is “accurate enough”. 8 bit quantization is ¼ of the original 32 bit, so it takes up ¼ of the memory. A 2 bit quant is 16 times smaller. Just like with compressing images, quality loss (measured in LLMs primarily via perplexity score) becomes noticeable at higher compression / smaller quants. A 4 bit quantization, often denoted as Q4, is usually the point after which quality drop becomes noticeable to human users.

Many users recommend using a smaller quant of a larger model rather than a larger quant of a smaller model. This depends on your use case. Narrow-trained smaller models can excel at tasks the larger model wasn’t trained for. For general use however, this concept holds, as larger models are “more resistant” to quantization. This is not technically true, as quantization degrades the model in an exponential fashion, however the quality of large models exceeds smaller models by enough that degradation takes longer to become noticeable. A general rule of thumb is that a Q2 quant of a model will have about the same perplexity as a Q8 quant of the next smaller model size, though this does not apply to 34B models, as the gap between a 34B and a 13B model is large enough that a 34B Q2 is still significantly better than a 13B Q8. See this post for details: https://github.com/ggerganov/llama.cpp/pull/1684.

**K quants** - a specific type of quantization used for GGUF files. It is adaptive, more heavily compressing less vital parts of the model while using less compression in more important parts. It does not play well with Mixtral models at the moment - use regular Q quants instead.

**S, M, and L (Quantization Mixes)** - different levels of quantization are used for different layers in the model.

S = same quant level used across the whole model. Results in maximum compression and quality loss.

M = Uses one level lower compression for the attention and feedforward layers. Slightly less compression, slightly lower quality loss than S.

L = Uses two levels lower compression for the attention and feedforward layers. SIightly less compression, slightly lower quality loss than M.

There are some exceptions to these rules. See the link ending in “1684” above for more details.

**BPW (bits per weight)** - quantization notation used mostly for EXL2 files. It tends to use more precise numbers, like 4.65, 5.0, etc. These translate roughly to the round numbers of Q quants.

**H number (head quantization)** - Similar to K quants, but for EXL2 files. Compresses only certain layers, typically attention heads, while leaving other layers less or uncompressed.

**QUIP#** (Quantization with Incoherence Processing) - a 2-bit quantization method that hugely reduces degradation. Uses incoherent matrices to reduce correlation between nodes, making each piece of information less related to the others, and therefore easier to compress. Incorporates lattice codebooks, which reduce information loss during compression. Also includes adaptive rounding to tailor the individual weights more intelligently. Essentially makes 2-bit quants worthwhile.

**SqueezeLLLM** - quantization method that intelligently applies dense and sparse quantization, incoherent matrices, lattice codebooks, and adaptive rounding (see above).

# File Types
**FP16** - the original, unquantized format of most models. Highest inference quality, massive resource use. 

**GGML (GPT-Generated Model Language)** - Original CPU/GPU split filetype. Not used much any more due to the development of GUUF.

**GGUF (GPT-Generated Unified Format)** - GPU/CPU split file type, successor to GGML. Extensible, unlike GGML, with better tokenization and support for special tokens. Self-contained, with recommended model and loader metadata in the file. Use if you can’t fit the whole model into VRAM. Comes in many quantizations. Load with llama.cpp.

**AWQ (Activation-aware Weight Quantization)** - Quantized GPU only filetype. Supposedly degradation from quantization is reduced with this filetype. Can be faster than GPTQ. Load with AutoAWQ. Only works with single-GPU setups currently.

**GPTQ (Generative Pretrained Transformer Quantization)** - Quantized GPU only filetype. Quality may be sub-par compared to GGUF. Not very popular since the advent of EXL2.

**EXL2 (ExLlamaV2)** - extremely fast GPU only filetype using the exllamav2 loader. Comes in various quants. Performance may degrade in zero-shot prompting compared to GGUF. Cannot be run on Tesla GPU architecture.

Performance of different quantization / file types is a subject of ongoing research and debate. Currently, it seems like GGUF is the absolute highest quality method, with moderate speed, depending on how many layers you can offload. EXL2 is narrowing the gap in terms of support and availability, but may still result in slightly lower quality outputs. Essentially: quality: GGUF, speed: EXL2.


# Modifying Models
**Base model** - the original model, usually developed by a large corporation due to their high compute requirements. Examples: Llama, Falcon, GPT-4, BART/BERT, T5, etc.

**Training** - the computational process that refines the full matrix of weights of the model. Models can be trained on a narrow set of data for speciality in specific tasks, or a broader range of data for a more all-around model. As mentioned above, if you have to use smaller models due to hardware constraints, select one that is trained on the type of tasks you want to use it for.

**LoRA (Low Rank Adaptation)** - a method of modifying a model without fully retraining it. Instead of modifying the entire weight matrix, it inserts small, low-rank "adapter" matrices at key points within the LLM architecture. These adapters have significantly fewer parameters (usually a few hundred MBs) compared to the full weight matrix. This requires much less time and compute to develop. LoRAs cannot be used between models, they are specific to the model they were trained with. This may be fixed with S-LoRA.

**QLoRA (Quantized Low Rank Adaptation)** - quantizes the model before training the LoRa weights. This creates a “double quant”, one for the model and one for the LoRA. Along with a few other features, this vastly reduces the resource demand for training and running the model and the LoRA.

**S-LoRA** - currently in development, allows LoRAs to be “hot-swapped” between models.

**Fine tune** - the process of adapting a pre-trained LLM to perform well on a specific task or domain. This is often a full retraining, and thus can be resource intensive.

**RoPE (Rotary Position Embeddings)** - instead of relating words in the context only to their neighbors, RoPE assigns two values to word placement: absolute position in the context and relative position compared to other words in the context. This additional data makes it easier for the model to recall information over longer contexts.

**YaRN** - RopE-style (see above) training method that extends the context of Llama 2 models to 128k tokens.

**RLHF (Reinforcement Learning through Human Feedback)** - Feedback is generated from human input. The data then can be incorporated into the model to guide its responses. Offers more nuanced feedback than DPO.

**DPO (Diredct Preference Optimization)** - Multiple response options are given, and the human user chooses their preference. The preferences are then reintegrated into the model to further guide responses.

**CALM (Composition to Augment Language Models)** - method of augmenting models that that essentially tacks a small model into a larger one. Useful for improving a specific domain in a model without degrading its skills in other domains.

**SLERP (Spherical Linear Interpolation)** - merging technique that provides smoother adjustments among weights, resulting in a more cohesive model with lower losses of defining characteristics.

**LASER (Layer-Selective Rank Reduction)** - model size reduction method developed by Microsoft that selectively prunes the attention and multilayer perceptron (not a typo) layers. Results in a smaller, faster model with less degradation than other pruning methods.

**DUS (Depth Up-Scaling)** - model layers are duplicated, pruned, pretrained, and then replace the original versions of the trained layers, resulting in additional model layers that improve performance. Developed as a way of enhancing small models without using MoE.

# Benchmarks
Tests used to empirically evaluate the model’s capabilities in various domains. Often used in training data, resulting in a perfect showcase of overfitting /  Goodheart’s Law: “ When a measure becomes a target, it ceases to be a good measure”.

**AI2 Reasoning Challenge (ARC)** - 25-shot test of grade-school science questions.

**HellaSwag** - 10-shot test of commonsense inference. Humans average 95%.

**MMLU** - 5-shot test of multitask accuracy, covering basic math, US history, computer science, law, and more.

**TruthfulQA** - 0-shot test to measure the model’s tendency to repeat common falsehoods.

**Winogrande** - 5-shot test for commonsense reasoning.

**GSM8K** - 5-shot test to evaluate the model’s skills in basic math and multi-step mathematical reasoning.

**Zero-shot prompting** - giving the model no examples and trusting it to figure out how to fulfill the request based on its own internal understanding of the prompt. This is the most difficult test for a model and often generates poor results, especially in smaller models.

**Single-shot** - the prompt contains a single example of the expected type of output.

**Few-shot** - 2-5 examples of the desired type of output are given in the prompt.

**Many-shot** - 5-20+ examples are given to the model.

# “Flavors” of Models
Some of these are base models, some of them are model modification techniques.

**LLaMA** - built on Google’s Transformer architecture. Very common and well integrated. Strong with factual tasks and question answering, weaker with creative tasks and coding. Base context of 4096 tokens.

**Falcon** - A modified transformer architecture improving performance and efficiency. Trained on curated dataset RefinedWeb. More creative than llama, but may be less factually accurate with reduced information retrieval performance. Not much innovation on this front due to usability issues, has fallen out of popularity.

**Mistral** - family of models developed by MistralAI. Known for their unusually high performance for their size due to the implementation of sparse transformers. Mistral was also the first to implement the mixture-of-experts (MoE) model architecture.

**MPT** - early entrants into the open source model arena. Haven’t come out with much lately and have faded into comparative obscurity.

**StarCoder** - family of models specializing in code generation. Less common.

**Replit** - family of coding models. Uncommon.

**GPT-Neo-X** - developed by EleutherAI. Uncommon.

**MoE (Mixture of Experts)** - an amalgamation of several (originally 8) smaller, specialized models that will be intelligently invoked by a filter depending on the task at hand. This allows for the capability (and memory requirements) of a larger model while only requiring the compute power of a smaller model. It is also less demanding to train.

**Mamba** - novel neural network architecture developed as an alternative to transformer models. Its greatest advantage is linear scaling with context, as opposed to exponential scaling in transformer models.  This allows for extremely long context windows with minimal degradation of performance. This is further enhanced by the use of selective state spaces, only selecting relevant context rather than reviewing the full context every time.

**WizardLM** - family of models trained with Evol-Instruct. Specializes in instruction following and code generation.

**REMM** - an older term referring to the MythoX family of models, 13B models known for their creativity and well-rounded abilities.

**Phi** - small models (~2B) developed by Microsoft. Generally thought to be on par with the performance of 7B models, which is impressive. Not good for chat, RP, storytelling, etc.

**Bagel** - models trained on extremely diverse training data, developed by Jon Durbin. Regular versions based on Yi, MoE versions based on Mistral/Mixtral.

**Alpaca** - llama based model developed for instruction-following by Stanford and Meta.

**Vicuna** - another llama based model, this one developed by Stanford and Meta. Trained on GPT-4 conversations.

**Guanaco** - llama based model trained with a 4-bit QLoRA on the OASST1 dataset.

**Solar** - ~11B model introduced by Upstage.  It uses depth-upscaling (DUS) to improve model performance with modest increase in model size.

**Yi** - 7B and 34B models developed by 01-AI. Known for their power and well-rounded abilities. Extremely popular.

**Airoboros** - family of models developed by Jon Durbin. Trained on synthetic instruction data derived from a heavily modified version of the self-instruct method of training.

**Zephyr** - a family of models that were some of the first to use DPO. Generally small, very capable models that focus on accuracy and helpfulness.

# Controlling Outputs with Parameters and Samplers
Most of this is taken from Text-Generation-Webui’s explanations here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab

**Logits** - un-normalized probabilities attached to each token. Normalized probabilities occur from 0 (never occurs) to 1 (always occurs).

**Temperature** - Broad, loose control of the randomness of the output. Higher temperature allows for more diversity and creativity, but can lead to incoherent outputs.

**Temperature last** - runs the temperature parameter last. Can produce better results than running first in some situations.

**Top P** - the model ranks all possible next tokens based on their normalized probability distribution. Top P selects a subset of these tokens with a cumulative probability that does not exceed P. Larger top P will allow for more creative and diverse outputs.

**Top K** - Like top P, but selects from the pool of tokens with K or greater probability. Lower top K makes text more coherent, but less creative.

**Min P** - Tells the model to disregard tokens with a probability less than P. Higher min P results in more coherent text while maintaining diversity more strongly than top P or top K.

**Mirostat + Tau** - Adjusts temperature on a per-token basis. Tau is the value that controls temperature. 8 is recommended value.

**Mirostat ETA** - ??? 0.1 is recommended value.

**Max new tokens** - The maximum number of new tokens to generate. If you are getting nonsense from your model after a certain length of reply, try shortening this.

**Min length** - minimum number of new tokens to generate. Ensures replies aren’t just an emoji or whatever.

**Repetition penalty** - How strongly to deter the model from repeating itself. Higher = less repetition. High rep penalty can create odd responses. Low rep penalty can lead to runaway repetition.

Note: If your model is repeating itself uncontrollably, you have probably asked it to create a response longer than any response it was trained on. Shorten your max new token length or pick a model trained on longer examples.

**Repetition penalty range** - how far back in the response to apply repetition penalty.

**Presence penalty** - Similar to repetition_penalty, but with an additive offset on the raw token scores instead of a multiplicative factor. It may generate better results. 0 means no penalty, higher value = less repetition, lower value = more repetition. Previously called "additive_repetition_penalty".

**Frequency penalty** - Repetition penalty that scales based on how many times the token has appeared in the context. Be careful with this; there's no limit to how much a token can be penalized.

**Typical P** - If not set to 1, select only tokens that are at least this much more likely to appear than random tokens, given the prior text.

**TFS** - Tries to detect a tail of low-probability tokens in the distribution and removes those tokens.

**Top A** - Tokens with probability smaller than (top_a) * (probability of the most likely token)^2 are discarded.

**Epsilon cutoff** - In units of 1e-4; a reasonable value is 3. This sets a probability floor below which tokens are excluded from being sampled.

**ETA cutoff** - In units of 1e-4; a reasonable value is 3. The main parameter of the special Eta Sampling technique.

**Guidance scale** - The main parameter for Classifier-Free Guidance (CFG). 1.5 is recommended value.

**enalty alpha** - Contrastive Search is enabled by setting this to greater than zero and unchecking "do_sample". It should be used with a low value of top_k, for instance, top_k = 4.

**Do sample** - When unchecked, sampling is entirely disabled, and greedy decoding is used instead (the most likely token is always picked).

**Encoder repetitions penalty** - Also known as the "Hallucinations filter". Used to penalize tokens that are not in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.

**No repeat ngram size** - If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases.

**Num beams** - Number of beams for beam search. 1 means no beam search. [What is beam search?]

**Length penalty** - Used by beam search only. Length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

**Early_stopping** - Used by beam search only. When checked, the generation stops as soon as there are "num_beams" complete candidates; otherwise, a heuristic is applied and the generation stops when is it very unlikely to find better candidates.

**DruGS (Deep Random micro-Glitch Sampling)** - Injects random noise into the inference as it passes through layers, making responses more creative and less deterministic. Still experimental - work is being done to find out how many and which layers produce optimal results with noise injection.

**Chunk tokens** - 

**Dynamic Temp** - 
