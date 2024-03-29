To jump to a section, click on the three dots and three bars icon to the right.


# General Terminology

**LLM (Large Language Model)** - type of artifical intelligence trained to perform natural language (human-like, not computer language) tasks.

**Inference** - What LLMs are designed to do - infer what words come next in a sequence. Can be done via CPU, GPU or a combination of both. GPU inference is roughly 10x faster than CPU inference, and any use of the CPU will greatly slow down inference speed. VRAM in consumer GPUs goes up to 24 GB currently. Many models are bigger than this, so CPU inference is required to run larger models on consumer hardware.

Note: if you have deep pockets, you can run multiple GPUs in parallel with minimal sacrifice to inference speed. Alternatively, you can pick up enterprise grade hardware. The current king of this is the NVIDIA H100 card with 80 GB of VRAM. They go for about $30k each. Good luck getting your hands on one.

**Model** - The neural network architecture designed to process and generate text. They are trained on a narrow or wide variety of subjects and tasks. It is best to select a model trained for what you would like to do with it. Have a collection of models for specific tasks, rather than trying to find an excellent all-around model, especially if you cannot easily run models larger than 13B.

**Parameters (B number)**- pieces of knowledge within the model’s training, usually in the billions range. The greater the number of parameters, the more the model “knows”, and so the better it will be at applying that knowledge in comprehensive, connective ways. Models with more knowledge tend to create “smarter” responses. More parameters also takes up more memory.

**Tokens** - the smallest chunk of text a model can process. Like syllables in language. A good rule of thumb is that for English, a token is 0.75 words. 4096 tokens, a common context length, will be about 3,072 words. 32k tokens will be about 24k words.

**Weights** - numerical values assigned to the edges (connections) and between neurons (nodes) within the neural network during training. These represent the strength of the connections, and are adjusted iteratively during training. The weights determine the influence of each connection on the final output and play a critical role in shaping the model’s behavior and performance. Raw, uncompressed weights consume a lot of memory, so ways to reduce this footprint allow for smaller, faster models at the cost of some degradation.

**Perplexity** - a score measuring uncertainty when the model is inferencing. Lower indicates better quality - generally, more intelligent and coherent responses from the model.

**Compute** - a term that in this context has come to refer to raw computing power. If something needs a lot of compute, say training a base model, it requires significant processing power in high density. With the price of enterprise grade hardware and its current necessity in developing AI, there is some concern regarding large corporation’s comparative “monopoly” on compute.

**Context window** - the amount of tokens the model can keep in its “short term memory”. Once the interaction has exceeded this number of tokens, the model will no longer be aware of their content. The model will also be unable to generate a coherent response longer than its context length. Context window is generally established in training, though it can be altered somewhat through other methods discussed further down.

Note: maximum context size of a model can be found in its config.json file under "max_position_embeddings".

**Tokens Per Second (TPS, t/s)** - the speed at which the model outputs tokens. 5 t/s is roughtly the lower limit for "live" feeling interactions like chat, slower than that, extended use gets tedious. Average reading speed is 8-12 t/s. Very large models or poorly optimized formats on weak hardware can drop speed to 0.5 t/s or less, and small, optimized models on strong hardware can reach 50+ t/s.

**Prompt** - the text you send to the model for it to inference from.

**Transformer** - type of neural network architecture that excels in language tasks. Most LLMs are currently based on transformer architecture.

**GPT (Generative Pretrained Transformer)** - type of transformer that is trained to predict next token(s) in a sentence.

**Pre-training** - to add.

**SOTA (State of the Art)** - A tag indicating a novel concept or application.



# Quantization
Most models are developed in 32-bit floating point (FP32) representation. This is extremely precise and captures nuanced data well, but also takes up a huge amount of space. A general rule of thumb is that 32 bit models require 4 GB of memory (RAM or VRAM) per 1 billion parameters, so a 7B model would require 28 GB or memory. This is not attainable on most consumer hardware, so models are quantized to reduce resource demand.

Quantization is basically compression, folding lots of individual pieces of information into a single piece of information that is “accurate enough”. 8 bit quantization is ¼ of the original 32 bit, so it takes up ¼ of the memory. A 2 bit quant is 16 times smaller. Just like with compressing images, quality loss (measured in LLMs primarily via perplexity score) becomes noticeable at higher compression / smaller quants. A 4 bit quantization, often denoted as Q4, is usually the point after which quality drop becomes noticeable to human users.

Many users recommend using a smaller quant of a larger model rather than a larger quant of a smaller model. This depends on your use case. Narrow-trained smaller models can excel at tasks the larger model wasn’t trained for. For general use however, this concept holds, as larger models are “more resistant” to quantization. This is not technically true, as quantization degrades the model in an exponential fashion, however the quality of large models exceeds smaller models by enough that degradation takes longer to become noticeable. A general rule of thumb is that a Q2 quant of a model will have about the same perplexity as a Q8 quant of the next smaller model size, though this does not apply to 34B models, as the gap between a 34B and a 13B model is large enough that a 34B Q2 is still significantly better than a 13B Q8. See this post for details: https://github.com/ggerganov/llama.cpp/pull/1684.

**K quants** - a specific type of quantization used for GGUF files. It is adaptive, more heavily compressing less vital parts of the model while using less compression in more important parts. It does not play well with Mixtral models at the moment - use regular Q quants instead. Recently updated to include an importance matrix, which reduces the perplexity increase of heavy quantization.

**S, M, and L (Quantization Mixes)** - different levels of quantization are used for different layers in the model.

- S = same quant level used across the whole model. Results in maximum compression and quality loss.

- M = Uses one level lower compression for the attention and feedforward layers. Slightly less compression, slightly lower quality loss than S.

- L = Uses two levels lower compression for the attention and feedforward layers. Slightly less compression, slightly lower quality loss than M.

- There are some exceptions to these rules. See the link ending in “1684” above for more details.

**BPW (bits per weight)** - quantization notation used mostly for EXL2 files. It tends to use more precise numbers, like 2.4, 4.65, etc. These translate roughly to the round numbers of Q quants.

**H number (head quantization)** - Similar to K quants, but for EXL2 files. Compresses only certain layers, typically attention heads, while leaving other layers less or uncompressed.

**QUIP# (Quantization with Incoherence Processing)** - a 2-bit quantization method that hugely reduces degradation. Uses incoherent matrices, lattice codebooks, and adaptive rounding. Essentially makes 2-bit quants worthwhile. See "Modifying Models" section for descriptions of these techniques.

**SqueezeLLLM** - quantization method that intelligently applies dense and sparse quantization, incoherent matrices, lattice codebooks, and adaptive rounding.

# File Types
**FP16** - the original, unquantized format of most models. Highest inference quality, massive resource use. Currently required for training LoRAs.

**GGML (GPT-Generated Model Language)** - Original CPU/GPU split filetype. Not used much any more due to the development of GUUF.

**GGUF (GPT-Generated Unified Format)** - GPU/CPU split file type, successor to GGML. Extensible, unlike GGML, with better tokenization and support for special tokens. Self-contained, with recommended model and loader metadata in the file. Use if you can’t fit the whole model into VRAM. Comes in many quantizations. Load with llama.cpp.

**AWQ (Activation-aware Weight Quantization)** - Quantized GPU only filetype. Supposedly degradation from quantization is reduced with this filetype. Can be faster than GPTQ. Load with AutoAWQ. Only works with single-GPU setups currently.

**GPTQ (Generative Pretrained Transformer Quantization)** - Quantized GPU only filetype. Quality may be sub-par compared to GGUF. Not very popular since the advent of EXL2.

**EXL2 (ExLlamaV2)** - extremely fast GPU only filetype using the exllamav2 loader. Comes in various quants. Performance may degrade in zero-shot prompting compared to GGUF. Cannot be run on Tesla GPU architecture.

**AQLM (Additive Quantization for Language Models** - a quantization method that uses additive vector quantization, a method that compresses values by merging mutual information between the quantized values. Reductions in perplexity under high quantization are greatest in smaller models, whereas larger models benefit less from this technique.

Performance of different quantization / file types is a subject of ongoing research and debate. Currently, it seems like GGUF is the absolute highest quality method, with moderate speed, depending on how many layers you can offload. EXL2 is narrowing the gap in terms of support and availability, but may still result in slightly lower quality outputs. Essentially: fidelity: GGUF, speed: EXL2.



# Modifying Models
**Base model** - the original model, usually developed by a large corporation due to their high compute requirements. Examples: Llama, Falcon, GPT-4, BART/BERT, T5, etc.

**Training** - the computational process that refines the full matrix of weights of the model. Models can be trained on a narrow set of data for speciality in specific tasks, or a broader range of data for a more all-around model. As mentioned above, if you have to use smaller models due to hardware constraints, select one that is trained on the type of tasks you want to use it for.

**Epoch** - one complete pass on the training data. Multiple epochs can improve retention of information, but too many can result in overfitting and inflexibility of the model. Experiment with different numbers of epochs to find the optimal number of passes. 1 to 3 epochs are common.

**LoRA (Low Rank Adaptation)** - a method of modifying a model without fully retraining it. Instead of modifying the entire weight matrix, it inserts small, low-rank "adapter" matrices at key points within the LLM architecture. These adapters have significantly fewer parameters (usually a few hundred MBs) compared to the full weight matrix. This requires much less time and compute to develop. LoRAs cannot be used between models, they are specific to the model they were trained with. This may be fixed with S-LoRA.

**QLoRA (Quantized Low Rank Adaptation)** - quantizes the model before training the LoRa weights. This creates a “double quant”, one for the model and one for the LoRA. Along with a few other features, this vastly reduces the resource demand for training and running the model and the LoRA.

**S-LoRA (Scalable Low Rank Adaptation)** - currently in development, allows LoRAs to be “hot-swapped” between models.

**RoSA (Robust Adaptation)** - parameter-efficient fine-tuning method that jointly trains low-rank and highly-sparse components on top of a set of fixed pretraining weights to efficiently approximate the performance of a full fine-tune. Outperforms LoRAs at the same parameter budget.

**PESC (Parameter-Efficient Sparsity Crafting)** - transitions dense models to sparse models using a Mixture of Experts (MoE) architecture. Essentially makes each expert a LoRA rather than a full model, significantly reduces memory requirements.

**Fine tune** - the process of adapting a pre-trained LLM to perform well on a specific task or domain. This is often a full retraining, and thus can be resource intensive.

- SFT (Supervised Fine Tuning) - fine tuning using human-validated training data.

- PEFT (Parameter-Efficient Fine-Tuning) - methods of fine-tuning without modifying all paramters. Reduces time and resource demand of fine-tuning.

- RLHF (Reinforcement Learning through Human Feedback) - Feedback is generated from human input. The data then can be incorporated into the model to guide its responses. Offers more nuanced feedback than DPO.

- Reward Models - essentially a "second layer" LLM trained to evaluate the outputs of the main LLM and assign rewards based on how desirable or acceptable those outputs are based on predefined criteria.

- DPO (Diredct Preference Optimization) - Multiple response options are given, and the human user chooses their preference. The preferences are then reintegrated into the model to further guide responses.

- PPO (Proximal Policy Optimization) - an algorithm often used in conjunction with reward models to train models. It drives the process of adjusting the LLM's behavior based on the guidance provided by the reward model. Often very sensitive to hyperparameter settings, and requires experimentationn to optimize.

- Reject sampling - a component of fine-tuning where outputs are evaluated and unsuitable low-quality ones are excluded based on predefined criteria and the better quality outputs are fed back into the fine-tuning process.

**RoPE (Rotary Position Embeddings)** - instead of relating words in the context only to their neighbors, RoPE assigns two values to word placement: absolute position in the context and relative position compared to other words in the context. This additional data makes it easier for the model to recall information over longer contexts.

**YaRN** - RopE-style (see above) training method that extends the context of Llama 2 models to 128k tokens.

**CALM (Composition to Augment Language Models)** - method of augmenting models that that essentially tacks a small model into a larger one. Useful for improving a specific domain in a model without degrading its skills in other domains.

**SLERP (Spherical Linear Interpolation)** - merging technique that provides smoother adjustments among weights, resulting in a more cohesive model with lower losses of defining characteristics.

**LASER (Layer-Selective Rank Reduction)** - model size reduction method developed by Microsoft that selectively prunes the attention and multilayer perceptron (not a typo) layers. Results in a smaller, faster model with less degradation than other pruning methods.

**DUS (Depth Up-Scaling)** - model layers are duplicated, pruned, pretrained, and then replace the original versions of the trained layers, resulting in additional model layers that improve performance. Developed as a way of enhancing small models without using MoE.

**Lattice codebooks** - model weights are intelligently grouped and assigned to a single numerical value and then stored in multi-dimensional grid (lattice). This compresses the weight file while with less degradation, reducing model size while minimizing loss.

**Incoherent matrices** - when the model's weights are assigned to a matrix during training, they are transformed to ensure that their coordiantes not extreme. This makes them better suited for adaptive rounding.

**Adaptive rounding** - intelligent method of rounding weights that adapts them to minimize the error rounding produces. This reduces the damaging effects of quantization.

**Brain Hacking Chip** - typical CFG only alters the logits that come out of the final layer. Brain hacking chip applies CFG to the output of every layer of the model as inference is running and applies customized CFG weights for each layer. Now comes with DRuGs. Introduced by SoylentMithril.



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
Some of these are base models, some of them are model modification techniques.\

**Chat** - a model that has been trained on chat-style interactions. Feels more conversational and can handle multi-trun interactions better.

**LLaMA** - built on Google’s Transformer architecture. Very common and well integrated. Strong with factual tasks and question answering, weaker with creative tasks and coding. Base context of 4096 tokens.

**Falcon** - A modified transformer architecture improving performance and efficiency. Trained on curated dataset RefinedWeb. More creative than llama, but may be less factually accurate with reduced information retrieval performance. Not much innovation on this front due to usability issues, has fallen out of popularity.

**Mistral** - family of models developed by MistralAI. Known for their unusually high performance for their size due to the implementation of sparse transformers. Mistral was also the first to implement the mixture-of-experts (MoE) model architecture.

**MPT** - early entrants into the open source model arena. Haven’t come out with much lately and have faded into comparative obscurity.

**StarCoder** - family of models specializing in code generation. Less common.

**Replit** - family of coding models. Uncommon.

**GPT-Neo-X** - developed by EleutherAI. Uncommon.

**MoE (Mixture of Experts)** - an amalgamation of several (originally 8) smaller, specialized models that will be intelligently invoked by a filter depending on the task at hand. This allows for the capability (and memory requirements) of a larger model while only requiring the compute power of a smaller model. It is also less demanding to train.

**Mamba** - novel neural network architecture developed as an alternative to transformer models. Its greatest advantage is linear scaling with context, as opposed to exponential scaling in transformer models.  This allows for extremely long context windows with minimal degradation of performance. This is further enhanced by the use of selective state spaces, only selecting relevant context rather than reviewing the full context every time.

**MambaByte** - a version of Mamba that learns from bytes rather than tokens like transformers do. Byte-level learning requires much longer sequences, but the state-space architecture can be modified to offset this hurdle. MambaByte has been found to be competitive with both trasformer models and other byte-level models.

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

**SUS** - family of Yi-based Chinese / English models developed in part by Southern University of Science and Technology, fine-tuned for complex instruction following. 8K context.

**XWIN** - family of models developed by XwinLM. Use supervised fine tuning, reward models, and RHLF to develop very creative, capable models.

**NoroMaid** - to add.

**DeepSeek** - family of models trained with self-supervized pretraining, supervised fine-tuning and DPO on 2 trillion tokens.



# Controlling Outputs with Parameters and Samplers
Most of this is taken from Text-Generation-Webui’s explanations here: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab

**Logits** - un-normalized probabilities attached to each token. Normalized probabilities occur from 0 (never occurs) to 1 (always occurs).

**Temperature** - Broad, loose control of the randomness of the output. Higher temperature allows for more diversity and creativity, but can lead to incoherent outputs.

**Temperature last** - runs the temperature parameter last. Can produce better results than running first in some situations.

**Top P** - the model ranks all possible next tokens based on their normalized probability distribution. Top P selects a subset of these tokens with a cumulative probability that does not exceed P. Larger top P will allow for more creative and diverse outputs.

**Top K** - Like top P, but selects from the pool of tokens with K or greater probability. Lower top K makes text more coherent, but less creative.

**Min P** - Tells the model to disregard tokens with a probability less than P. Higher min P results in more coherent text while maintaining diversity more strongly than top P or top K.

**Mirostat + Tau** - Adjusts temperature on a per-token basis. Tau is the value that controls temperature. 8 is recommended value.

**Mirostat ETA** - ??? 0.1 is recommended value. to add.

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

**Penalty alpha** - Contrastive Search is enabled by setting this to greater than zero and unchecking "do_sample". It should be used with a low value of top_k, for instance, top_k = 4.

**Do sample** - When unchecked, sampling is entirely disabled, and greedy decoding is used instead (the most likely token is always picked).

**Encoder repetitions penalty** - Also known as the "Hallucinations filter". Used to penalize tokens that are not in the prior text. Higher value = more likely to stay in context, lower value = more likely to diverge.

**No repeat ngram size** - If not set to 0, specifies the length of token sets that are completely blocked from repeating at all. Higher values = blocks larger phrases, lower values = blocks words or letters from repeating. Only 0 or high values are a good idea in most cases. Essentially prevents repetition of full phrases, incluing from the prompt. Prevents parroting.

**Num beams** - Number of beams for beam search. 1 means no beam search. [What is beam search?]

**Length penalty** - Used by beam search only. Length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

**Early_stopping** - Used by beam search only. When checked, the generation stops as soon as there are "num_beams" complete candidates; otherwise, a heuristic is applied and the generation stops when is it very unlikely to find better candidates.

**DruGS (Deep Random micro-Glitch Sampling)** - Injects random noise into the inference as it passes through layers, making responses more creative and less deterministic. Still experimental - work is being done to find out how many and which layers produce optimal results with noise injection.

**Chunk tokens** - how many tokens to process at once. More tokens allows the model to consider a wider context at the cost of greater resource usage. Low values will significantly degrade the quality of outputs. Very high values can overwhelm the model's attention, causing it to miss or forget things.

**Dynamic Temp** - Adjusts the temperature proportionally to the standard deviation of the sampled set of tokens. When there are many viable tokens, the model will select tokens of lower probability. When there are few viable tokens, the model will select higher probability tokens. Allows for more creative responses while still maintaining coherence.

**Special tokens** - denote structure and organization of inputs, direct the model towards certain tasks or behaviors, etc. Essentially brief annotations to help the model understand how to view and use the input.

**CFG (Classifier-Free Guidance)** - to add.



# Instruction Formats
The format in which instruction-following training data was appended. This whole thing is a mess, and is in dire need of standardization / unification.

Using a format the model wasn’t trained for will produce poor outputs.

I really don’t want to sort through this crap. I’m not even sure how to use instruction formats properly. A reddit post that details some of it can be found in the second half of this post: https://www.reddit.com/r/Oobabooga/comments/19480dr/how_to_train_your_dra_model/

# General Memory Requirements
This can be RAM, VRAM, or a sum of the two.
Q4 + 4K context. Approx. values.

| Parameter Count | Memory Use |
| --------------- | ---------- |
| Tiny (<2.5B) | 2 GB |
| 3B | 4 GB |
| 7B | 8 GB |
| 13B | 16 GB |
| ~30B | 24 GB |
| 70B | 64 GB |
| 120B | 128 GB |
| 180B | 192 GB |



# Layers
Different components within the model architecture. I’m not super familiar with this stuff, and it’s not terribly important to know if you just want to play around.

**Embedding layer** - where the model interprets and captures the semantic meaning of the input, basically establishing context.

**Feedforward layer** - directs contextualized input through the model appropriately.

**Attention layers** - contain the Q, K, and V layers, explained below. Selectively examines input and knowledge for relevance.

**Recurrent layer** - incorporates input with knowledge via looping.

**Transformer** - translates information from one internal representation to another so that it can be used by various other layers.

**Q (Query) layer** - represents the information that a particular element in the sequence is looking for from other elements. Imagine it like highlighting keywords based on the word itself and its surrounding context.

**K (Key) layer** - holds the information what information other elements in the sequence can provide in response to the query. Acts as a context label.

**V (Value) Layer** - This layer contains the actual content that will be shared in response to the query. Like a detailed definition or explanation behind the key.



# Loaders
**Transformers** - loads FP16 or FP32 models. More detail here: https://github.com/oobabooga/text-generation-webui/wiki/04-%E2%80%90-Model-Tab#transformers

**Llama.cpp** (GGUFs) chances are good you're using this.

- N-gpu-layers - number of layers to offload to GPU to accelerate CPU inference. 0 = CPU only.

- N_ctx - context length of the model. Usually predetermined, but may need to be set according to the model loaded occasionally.

- Threads - number of CPU threads to use. Set to 1 if all layers have been loaded into GPU. Otherwise, set to number of physical cores your CPU has.

- Threads_batch - number of threads for batch processing. Set to total number of physical and virtual cores your CPU has.

- N_batch - batch size for prompt processing. 

- Alpha _value - extends context length of a model with minor quality loss. 1.75 = 1.5x context, 25 = 2x context length.

- Rope_freq_base - similar to alpha value, extends context in older models like CodeLlama that have been finetuned with longer contexts.

- Compress_poss_emb - original method for extending context length. Straight across multiplier. Use only with models that have been fine tuned with this parameter adjusted.

- Tensorcores - utilize Nvidia tensor cores to accelerate inference. Provides small gain in speed.

- No_offload_kqv - does not offload the K, Q, V to the GPU. Saves VRAM but reduces performance

- No_mul_mat_q - disables the multi matrix kernels. Likely reduces inference quality, especially at long contexts. Not sure why this would need to be used.

- No-mmap - loads the model into memory at once, possibly preventing I/O operations later on at the cost of a longer load time.

- Mlock - force the system to keep the model in RAM rather than swapping or compressing. Unsure of why this would be used.

- Numa - non-uniform memory access. May accelerate inference on multi-cPU systems.

- cpu - force CPU-compiled version of llama.cpp that uses CPU only. Activated if llama.cpp doesn’t work otherwise.

- Tensor_split - if you have multiple GPUs, sets the amount of memory to allocate per GPU.

- No_use_fast - disables fast version of tokenizer. Use only if the tokenizer for the model doesn’t work at first.

- Disable_exllama - use when loading GPTQ models with transformers loader. They will not work otherwise.

- Cache_8bit - create 8-bit precision cache instead of 16-bit. Saves VRAM but increases perplexity. Use allows for very long contexts lengths more easily.

**HF loaders** - Similar to non-HF loaders, but with transformers samplers, and using the transformers tokenizer instead of the internal llama.cpp tokenizer.

**ExLlama** - to add.

**ExLlamav2** - loads GPTQ and EXL2 models.

**AutoGPTQ** - to add.

**AutoAWQ** - loads AWQ models. See link above.

**GPTQ-for-Llama** - loads GPTQ models. See link above.

**cTransformers** - loads GGUF and GGML models. Great range of compatibility if your model doesn't work with transformers loader.

**QUIP# (Quantization with Incoherence Processing)** - to add.

**HQQ** - to add.



# Common Training Data Sets
Used to train models on specific tasks, contexts, and knowledge. Some of these overlap with the “Flavors of Models” and “Benchmarks” section.

**ai2_arc** - Abstraction and reasoning examples. Increases “intelligence” perceived in the model.

**Airoboros** - Variety of categories of synthetic instructions generated by GPT-4. Increases instruction-following ability with a heavily modified version of self-instruct.

**Apps** - Python coding set with 10K problems. Increases python coding ability.

**Belebele** - Multi-lingual reading comprehension. Increases model’s skills in non-english interactions.

**Bluemoon** - Roleplay data scaped from Bluemoon, then cleaned and formatted as ShareGPT. Improves roleplay ability.

**Boolq** - Selection of yes/no questions. Improves definitiveness of yes/no answers, a common problem with LLMs.

**Capybara** - 10,000+ examples of single- and multi-turn synthetic conversation. Improves model’s chat abilities.

**Cinematika** - RP-style data synthesized from movie scripts. Makes model more conversational and less rigid in answer style.

**Dolphin** - an open source recreation of Microsoft’s Orca dataset. Alignment and bias have been filtered out, fostering greater compliance in models.

**Drop** - reading comprehension improvement set.

**Emobank** - Emotion annotations using the Valence-Arousal-Dominance scheme. Improves emotional intelligence.

**Evol-Instruct** - training tool that generates complex instruction sets and appropriate responses. Used to improve instruction-following.

**Gutenberg** - Books and plain text designed to make the model more knowledgeable about public domain literature. Apparently it makes the model more interesting.

**LMSys Chat 1M** - Chats collected from GPT-4’s winning chats in Chatbot Arena. A DPO tuning set, guides model responses in ways preferred by humans.

**MathInstruct** - Composit dataset with a variety of math-related tasks and problem/question formats. Increases model’s skill in answering math questions.

**MMLU** - Massive Multitask Language Understanding - a wide variety of questions about various subject matters. Often used as a benchmark dataset. Introduces contamination that must be scrubbed for benchmarking. Improves many varieties of english and non-english tasks.

**Natural Instructions** - millions of instructions from 1600+ task categories. Improves many types of instruction following.

**NoroMaid** - family of roleplay-focused models trained on Aesir Private RP dataset.

**OASST1** - dataset of human-generated and annotated assistant-style conversations. Makes the model more helpful.

**OpenBookQA** - Question answering improvement dataset.

**OpenOrca** - A collection of GPT3.5/4 chat completions. Improves chat performance.

**Orca** - Dataset developed by Microsoft designed to enhance small-model reasoning capabilities.

**Pippa** - Personal Interaction Pairs between People and AI. 26,000 conversations between users and bots on Character.AI. Improves roleplay.

**Piqa** - improves common sense reasoning in models.

**Puffin** - The first dataset/model released by Nous Research. 13B.

**Python-Alpaca** - python instruction response pairs, verified as functional. Improves python coding with Alpaca format.

**Rosetta-Code** - Code problems and solutions in a variety of programming languages. Improves coding abilities.

**Spider** - SQL targeted dataset. Improves model’s use of SQL.

**Squad V2** - Contextual question answering. Improves RAG performance.

**Synthia** - GPT-4 generated data using advanced prompting. No description of benefits offered.

**Winogrande** - Fill in the blank style prompts. Improves common sense reasoning.

**Toxic** - toxic and illegal content designed to remove censorship from the model.



# Notable Players in the LLM Space


**Individuals:**

**Eric Hartford** - AI researcher and curator of Cognitive Computations. Helped develop the Dolphin and Samantha models.

**Jon Durbin** - AI researcher, developed models like Airoboros and Bagel.

**Maxime LaBonne** - LLM researcher, keeps a very useful blog about LLM stuff. Created the Phixtral and some of the Neural models.

**Tom Jobbins / TheBloke** - Quantizes thousands of models on HF at an alarming rate. GGUF, GPTQ, AWQ, etc.

**Teknium** - Cofounder of Nous Research. Aids in the development of the Nous Hermes models, among others.

**Tim Dettmers** - deep learning researcher who helped develop the Guanaco models. Check out his blog for useful articles about hardware, etc.

**u/WolframRavenwolf** - Reddit user who conducts extensive testing and ranking of models, particularly for roleplay use.

**u/FPHam** - Reddit, known as FartyPants on GitHub. Has developed several useful Oobabooga extensions and novelty models on HF.

**LoneStriker** - Known for quantizting hundreds of models in various levels of EXL2.

**Gryphe** - developed the popular MythoMax models, as well as others. He focuses on roleplay and creative models that tend to perform very well.


**Organizations:**

**Nous Research** - AI research firm. Developed Hermes, YaRN, Capybara, Puffin, and Obsidian models.

**Mistral AI** - developed the Mistral models and the MoE format.

**Stability AI** - developed the Beluga and Zephyr models.

**Meta** - Developed the Llama and Llama 2 models, the most widely used model base.

**Upstage** - AI research firm that developed the currently popular SOLAR models. They also develop other tools and systems of AI research and learning.

**Cognitive Computations** - AI research group founded by Eric Hartford. Developed the Dolphin, Samantha, and WizardLM models. 

**Microsoft** - You should know who this is by now. Developed Bing Chat (formerly Sydney) model based off of GPT-4. They have a controlling stake in OpenAI.

**OpenAI** - The company that started the LLM craze with the release of their ChatGPT model in November 2022.

**Qwen** - Chinese company that developed the Qwen models, known for their capability and censorship. Also develop multimodal models.

**01-AI** - Chinese startup that developed the Yi models, known for their capability and long context length.

**MosaicML** - Machine learning corporation that developed the MPT models.

**Replit** - to add.

**EleutherAI** - to add.

**LMSys** - to add.

**XwinLM** - to add.

**Andreessen-Horowitz (a16z)** - venture capital agency that focuses on technological advancement. Funds a lot of GPU training time for the open source community.

**HuggingFace** - The largest repository for open-source AI software. Where most models are made available.

**GitHub** - developer platform for sharing, collaborating on, and developing code projects. Most UIs and other AI tools can be found here.



# Augmenting Models
Ways of improving or changing a model's capabilities or performance without modifying the model itself.

**RAG (Retrieval Augmented Generation)** - documents and other files of reference are uploaded and converted into a vector database. The model is then enabled to search this database and extract relevant information, which is then incorporated into the response. Helps to increase the model's factual accuracy or add information to the model without training on it.

- Vector Databases - documents and other source material are analyzed for semantic relationships. These relationships are then stored in a vector database, making relevant context easier and faster to retrieve.

**Speculative Decoding** - enhances LLM efficiency by leveraging a small, fast drafting model that generates an output to be reviewed by a larger, slower, but more "skilled" model. This reduces the total compute needed for the model(s).

**Cascade Speculative Drafting** - improves the speed of speculative decoding by replacing the autoregressive generation in speculative decoding with horizontal and vertical cascades.

- Vertical Cascade - process in which an extremely small, efficient model, usually a statistical language model, generates the first draft and sends it to a larger model for review. The larger model then reviews the draft and adds to it, then presenting its draft to a yet larger model. This is done multiple times before the largest model outputs its generation. This technique accelerates generation.

- Horizontal Cascade - preceding tokens for the drafts are generated by larger models and are likely to be more effective. Succeeding tokens are generated by smaller models. This ensures that the tokens are less likely to be rejected, reducing autoregression in the model. This technique reduces slowdowns in generation.



# Implementation - UIs, backends, etc.
Not comprehensive, just a list of the major players.

To add.



# Inbox / To Be Categorized
I haven't looked into these yet.

**Grammar** - to add.

**Vocab / size** - to add.

**Voxta** - to add.

**VaM** - to add.

**BLOOM** - to add.

**vLLM** - to add.

**Triton** - to add.

**Z-loss** - to add

**MLC LLM / Vulkan** - to add.

**RWKV** - to add.
