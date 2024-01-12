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
