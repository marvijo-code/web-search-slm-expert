# Tiny from-scratch LLM: request to JSON tool call (research, 2026-10-21)

## Topic
A decoder-only transformer of 1-5M parameters, trained from random weights on synthetic templated data. It maps "go to example.com and give me the data" to {"url":"https://example.com","extract":[...]} for Puppeteer. Inference is plain JavaScript on a CPU laptop. [n] = source number below.

## 1. Canonical recipes (one line each)
- Vaswani 2017: the transformer block, scaled dot-product and multi-head attention [1].
- GPT-2: decoder-only LM, byte-level BPE (50,257 vocab), layer norm moved to the input of each sub-block (pre-norm), 1024 context [2].
- "Let's build GPT" video: a char-level GPT on tiny Shakespeare, built line by line in about 2 hours [3].
- minGPT: the readable PyTorch GPT. It is semi-archived. Its `projects/adder` and sort demo show tiny GPTs learning algorithmic tasks [4].
- nanoGPT: the training loop and configs. Its README now calls it deprecated and points to nanochat [5].
- llm.c: GPT-2/3 training in raw C/CUDA, no PyTorch [6].
- nanochat: the full pipeline (tokenizer, pretrain, finetune, eval, inference, web UI). One `--depth` dial sets the other hyperparameters [7].
- llama2.c: a Llama-style model with inference in one C file. This is the closest template for a JS port: KV cache, an int8 variant (runq.c), and a 260K-parameter TinyStories model [8].
- Raschka book/repo: tokenization, attention, GPT, pretraining, classification and instruction finetuning, LoRA; bonus GPT-to-Llama conversion [9].

## 2. Architecture choices at 1-5M parameters
- Pre-norm: keep it. It gives well-behaved gradients at init and needs less warmup [10]. It matters the most for stable training.
- RMSNorm: drops LayerNorm's mean-centering, 7-64% faster, equal quality [11]. Minor at this size, but simpler to port to JS.
- RoPE: relative position through rotation [12]. Useful for copying: induction heads match on "token after a match", which is a relative pattern. A learned absolute embedding also works at a fixed 256 context.
- SwiGLU: a GLU variant beats ReLU/GELU [13]. MobileLLM measured +1.3 points at 125M [15]. Small gain here; optional.
- Weight tying: tie input and output embeddings [14]. MobileLLM: embeddings are over 20% of a 125M model, and tying cost 0.2 points [15]. Here it depends on vocab. With a char vocab of about 100, embeddings are about 1% of parameters, so tying hardly matters. With an 8K BPE vocab at d=192, embeddings are about 1.5M parameters, half the model, so tie.
- No bias: a simplification taken from Llama-style models. Negligible effect at this scale.
- Depth over width: MobileLLM found deeper and thinner wins at 125M [15]. TinyStories found depth helps context tracking and width helps knowledge [16]. Copying is a context task, so lean deep.

## 3. Tokenizer
Use character or byte level (vocab about 100-260). Reasons:
- The model must copy arbitrary, never-seen domains. An induction head does [A][B]...[A] -> [B] [17]. At char level, copying is the same operation for every domain.
- BPE splits an unseen domain into pieces that can differ between prompt and output (" example" vs "//example"). The model then has to re-spell, not copy.
- ByT5: byte models are more robust to noise and better on spelling-sensitive tasks, at the cost of longer sequences [18].
- Transformers learn exact copying fast; fixed-state models do not [19].
- The cost of char level: about 80 output characters per JSON. At this model size that is still well under a second (section 7).

## 4. Evidence that tiny models can do narrow tasks
- TinyStories: models under 10M parameters write fluent stories. At about 2.5M (8 layers) text is coherent. At about 1M it is only grammatical. Diversity came from 3 random words plus random features per prompt [16][20].
- Tool calling: Toolformer (self-supervised API-call insertion) [21]. Gorilla: a finetuned LLaMA beats GPT-4 on API calls [22]. TinyAgent: a 1.1B model matches GPT-4-Turbo on a narrow function-calling set [23].
- Synthetic data: Self-Instruct = generate, then filter invalid and near-duplicate samples [24].
- Apply this here: randomize every slot (verbs, filler, domain shape, TLD, subdomain, path, casing, punctuation). Split held-out data by template and domain, never by sample.
- Caveat: all of these tool-calling models are 1B+ and pretrained. None is a 3M model from scratch. The from-scratch evidence is TinyStories and the minGPT algorithmic demos [4][16].

## 5. Hyperparameters
- AdamW. GPT-3: betas (0.9, 0.95), weight decay 0.1, clip 1.0, warmup then cosine to 10% of peak [25].
- nanoGPT char "baby GPT": 6 layers, 6 heads, d=384, ctx 256, batch 64, LR 1e-3 to 1e-4, warmup 100, 5000 iters, beta2 0.99, dropout 0.2 (the dataset is small, so it overfits) [26].
- llama2.c: small nets tolerate LR 1e-3 or higher [8].
- Chinchilla: scale parameters and tokens equally [27]. The popular reading is about 20 tokens per parameter (60M tokens for 3M). That is a compute-optimal rule for broad pretraining. It does not tell you when a narrow synthetic task is solved, and over-training a small model to make inference cheap is normal. Stop when held-out exact match plateaus.

## 6. Evaluation
- Exact match after `JSON.parse` plus key-order-free comparison.
- Report these separately: JSON validity rate, URL-field accuracy, extract-list accuracy.
- The held-out set has unseen TLDs, domain lengths, subdomains, paths and ports, unseen paraphrase templates, and negatives.
- Constrained decoding is the safety net: llama.cpp GBNF [28], Outlines FSM/CFG indexing [29].
- Is it cheating? Tam et al. found format restrictions hurt reasoning [30]. The dottxt team showed that this came from unequal prompts and parsing, and structured output scored equal or better [31]. Neither paper is about a 3M-parameter model.
- Verdict: not cheating if raw and constrained numbers are both reported. A grammar guarantees syntax, not that the right domain was copied.

## 7. CPU inference in JS
- KV cache plus greedy argmax, as in llama2.c run.c [8].
- llama2.c runs a 15M model at about 110 tok/s on an M1 in C fp32 [8].
- My estimate, not measured: a 3M model needs about 6 MFLOP/token, so roughly 150-500 tok/s in Float32Array JS. An 80-char reply takes well under 1 s.
- Store weights as int8 (Q8_0-style [8]) or fp16 for a 3-6 MB file, then dequantize to fp32 at load.
- Under a grammar, forced characters (`{"url":"https://`) skip sampling and can be fed as a batched prefill.

## Recommended approach
| Item | Value |
|---|---|
| Tokenizer | char/byte, vocab about 100 printable ASCII + BOS/SEP/EOS |
| Layers / d_model / heads | 6 / 192 / 6 (head dim 32) |
| FFN | SwiGLU, hidden 512 |
| Blocks | pre-norm RMSNorm, RoPE, no bias, tied embeddings |
| Size | about 2.7M parameters |
| Context | 256 |
| Loss | only on the JSON part |
| Optimizer | AdamW (0.9, 0.95), wd 0.1, clip 1.0, dropout 0 (data is unlimited) |
| LR | peak 1e-3, 200 warmup steps, cosine to 1e-4 |
| Batch | 64 x 256 |
| Steps | 10-20K (about 160-330M chars) |
| Data | 200K-1M fresh samples |

- Train on a free GPU notebook (Kaggle/Colab), not the laptop CPU.
- Ship with an exact-match gate of at least 99% on unseen domains and templates, raw (no grammar).

## Disagreements and caveats
- Constrained decoding helps or hurts: [30] vs [31].
- Cosine vs linear decay: GPT-3 uses cosine [25]. Cerebras-GPT preferred linear (seen in search, not verified).
- Chinchilla constants do not transfer across setups [27].
- nanoGPT is deprecated in favour of nanochat [5]. nanochat uses BPE and targets GPUs [7], so take its loop, not its tokenizer.
- Baseline to beat: a regex can pull the domain. The model's real value is mapping phrasing to the extract fields. Test against that baseline honestly.
- Unverified from primary source this pass: exact GPT-2 sections (the PDF did not render) and TinyStories hyperparameters beyond the sizes listed.

## Sources
1. Attention Is All You Need - https://arxiv.org/abs/1706.03762
2. Language Models are Unsupervised Multitask Learners (GPT-2) - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
3. Let's build GPT: from scratch, in code, spelled out - https://www.youtube.com/watch?v=kCc8FmEb1nY
4. karpathy/minGPT - https://github.com/karpathy/minGPT
5. karpathy/nanoGPT - https://github.com/karpathy/nanoGPT
6. karpathy/llm.c - https://github.com/karpathy/llm.c
7. karpathy/nanochat - https://github.com/karpathy/nanochat
8. karpathy/llama2.c - https://github.com/karpathy/llama2.c
9. rasbt/LLMs-from-scratch (Build a Large Language Model (From Scratch)) - https://github.com/rasbt/LLMs-from-scratch
10. On Layer Normalization in the Transformer Architecture - https://arxiv.org/abs/2002.04745
11. Root Mean Square Layer Normalization - https://arxiv.org/abs/1910.07467
12. RoFormer: Enhanced Transformer with Rotary Position Embedding - https://arxiv.org/abs/2104.09864
13. GLU Variants Improve Transformer - https://arxiv.org/abs/2002.05202
14. Using the Output Embedding to Improve Language Models - https://arxiv.org/abs/1608.05859
15. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases - https://arxiv.org/abs/2402.14905 (details: https://arxiv.org/html/2402.14905v2)
16. TinyStories (full text) - https://ar5iv.labs.arxiv.org/html/2305.07759
17. In-context Learning and Induction Heads - https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html (arXiv: https://arxiv.org/abs/2209.11895)
18. ByT5: Towards a token-free future with pre-trained byte-to-byte models - https://arxiv.org/abs/2105.13626
19. Repeat After Me: Transformers are Better than State Space Models at Copying - https://arxiv.org/abs/2402.01032
20. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? - https://arxiv.org/abs/2305.07759
21. Toolformer: Language Models Can Teach Themselves to Use Tools - https://arxiv.org/abs/2302.04761
22. Gorilla: Large Language Model Connected with Massive APIs - https://arxiv.org/abs/2305.15334
23. TinyAgent: Function Calling at the Edge - https://arxiv.org/abs/2409.00608
24. Self-Instruct: Aligning Language Models with Self-Generated Instructions - https://arxiv.org/abs/2212.10560
25. Language Models are Few-Shot Learners (GPT-3, Appendix B) - https://arxiv.org/abs/2005.14165
26. nanoGPT train_shakespeare_char.py - https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py
27. Training Compute-Optimal Large Language Models (Chinchilla) - https://arxiv.org/abs/2203.15556
28. llama.cpp GBNF grammars README - https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
29. Efficient Guided Generation for Large Language Models (Outlines) - https://arxiv.org/abs/2307.09702
30. Let Me Speak Freely? Impact of Format Restrictions on LLM Performance - https://arxiv.org/abs/2408.02442
31. Say What You Mean: A Response to 'Let Me Speak Freely' (dottxt) - https://blog.dottxt.ai/say-what-you-mean.html
