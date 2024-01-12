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
