# pdf-vlm

pdf-vlm is a tool designed to enable multi-modal summarization of PDF documents using vLLM. 

By converting each page of a PDF into an image, pdf-vlm sends batches of images to a vLLM server hosting a multi-modal model (such as Pixtral-12B-FP8). 
The server processes these images and generates preliminary JSON summaries for each chunk, which are later aggregated into a final, coherent summary of the document. 
This approach allows users to extract structured insights from long-form PDFs — combining visual and textual information for efficient content analysis.

The key tuning parameters are:

* On the vLLM server, set `--limit-mm-per-prompt` image=5 to control the maximum number of image inputs per prompt.
* In the client, set `--images_per_chunk 5` to define how many PDF pages are processed together in each summarization request.

## Setup

Install [vLLM](https://github.com/vllm-project/vllm) and install [pdf2image](https://github.com/Belval/pdf2image?tab=readme-ov-file#how-to-install)

```
sudo apt install poppler-utils
pip install -r requirements.txt
```

## Example Usage

Here is an example of using vLLM hosting Pixtral-12B-FP8 on 1xH100 to produce a summary of the paper ["Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"](https://arxiv.org/abs/2502.11089)

**Server:**
```
vllm serve nm-testing/pixtral-12b-FP8-dynamic --limit-mm-per-prompt image=5 --max-model-len 50000
```

**Client:**
```
python client.py data/2502.11089.pdf --endpoint http://localhost:8000/v1 --model nm-testing/pixtral-12b-FP8-dynamic --images_per_chunk 5
```

**Client Output:**

Converted PDF to 24 pages.

Processing pages 1 to 5...
Chunk summary:
```json
{
  "title": "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention",
  "authors": [
    "Jingyang Yuan",
    "Huazhu Gao",
    "Damai Dai",
    "Jureya Luo",
    "Liang Zhao",
    "Zhengyan Zhang",
    "Zhenda Xie",
    "T. X. Wei",
    "Lean Wang",
    "Zhiping Xiao",
    "Yuqiang Wang",
    "Chong Ruan",
    "Ming Zhang",
    "Wenfeng Liang",
    "Wangding Zeng"
  ],
  "abstract": "Long-context modeling is crucial for next-generation language models, yet the high computational cost of standard attention mechanisms poses significant challenges. Sparse attention offers a promising direction for improving efficiency while maintaining model capabilities. We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision. Our approach advances sparse attention design with two key innovations: (1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design, with implementation optimizations for modern hardware. (2) We enable end-to-end training, reducing pretraining computation without sacrificing model performance. As shown in Figure[1] experiments show the model pretrained with NSA maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning. Meanwhile, NSA achieves substantial speedups over Full Attention on 64k-length sequences across decoding, forward propagation, and backward propagation, validating its efficiency throughout the model lifecycle.",
  "sections": {
    "1. Introduction": "The research community increasingly recognizes long-context modeling as a crucial capability for next-generation large language models, driven by diverse real-world applications ranging from in-depth reasoning (DeepSeek-AI [2025], Zelikman et al. [2022]), repository-level code generation (Zhang et al. [2023a], Zhang et al.), and multi-turn autonomous agent systems (Park et al. [2023]). Recent breakthroughs, including OpenAI’s o-series models, DeepSeek-RI (DeepSeek-AI [2023]), and Gemini 1.5 Pro (Google et al. [2024]), enabling models to process entire codebases, lengthy documents, maintain coherent multi-turn conversations over thousands of tokens, and perform complex reasoning across long-range dependencies. However, the high complexity (Zheng et al. [2023]) of vanilla Attention (Vaswani et al. [2017]) mechanisms emerges as a critical latency bottleneck as sequence length increases. Theoretical estimates indicate that attention computation with softmax architectures accounts for 70–80% of total latency when decoding 64k-length contexts, underscoring the urgent need for more efficient attention mechanisms.",
    "Figure 1": "Comparison of performance and efficiency between Full Attention model and our NSA. Left: Despite being sparse, NSA surpasses Full Attention baseline on average across general benchmarks, long-context tasks, and reasoning evaluation. Right: For 64k-length sequence processing, NSA achieves substantial computational speedup compared to Full Attention in all stages: decoding, forward propagation, and backward propagation.",
    "Figure 2": "Overview of NSA’s architecture. Left: The framework processes input sequences through three parallel attention branches. For a given query, preceding keys and values are processed into compressed attention for coarse-grained patterns, selected attention for important token blocks, and sliding attention for local context. Right: Visualization of different attention patterns produced by each branch. Green areas indicate regions where attention scores need to be computed, while white areas represent regions that can be skipped."
  }
}
```

Processing pages 6 to 10...
Chunk summary:
```json
{
  "title": "Overall Framework and Algorithm Design",
  "authors": [],
  "abstract": "The document discusses a framework for optimizing attention mechanisms in neural networks. It proposes replacing key-value pairs with a more compact and information-dense set of representation key-value pairs, dynamically constructed based on the current query and contextual memory. The framework includes strategies for compression, selection, and sliding window, aiming to maintain a high sparsity ratio and reduce computational burden.",
  "sections": [
    {
      "title": "Overall Framework",
      "content": "To leverage the potential of attention with natural sparse pattern, we propose replacing the original key-value pairs k_q, v_q in Equation (1) with a more compact and information-dense set of representation key-value pairs k_t, v_t, given each query q_t. Specifically, we formally define the optimized attention output as follows: k_t = f_k(q_t, k_q, v_q), v_t = f_v(q_t, k_q, v_q). o_t^q = Attn(q_t, k_t, v_t), where k_t, v_t are dynamically constructed based on the current query q_t and the contextual memory k_q, v_q. We can design various mapping strategies to get different categories of k_t, v_t and combine them as follows: o_t^q = ∑_c∈C g_c^q * Attn(q_t, k_t^c, v_t^c). As illustrated in Figure[2], NSA have three mapping strategies C = {cmp, slc, win}, representing compression, selection, and sliding window for keys and values. g_c ∈ [0,1] is the gate score for corresponding strategy c, derived from input features via an MLP and sigmoid activation. Let N_t denote the total number of remapped keys/values: N_t = ∑_c∈C size[k_t^c]. We maintain a high sparsity ratio by ensuring N_t << t."
    },
    {
      "title": "Algorithm Design",
      "content": "In this subsection, we introduce the design of our remapping strategies f_k and f_v: token compression, token selection, and sliding window. 3.3.1. Token Compression By aggregating sequential blocks of keys or values into block-level representations, we obtain compressed keys and values that capture the information of the entire block. Formally, the compressed key representation is defined as: k_t^cmp = f_k^cmp(k_q) = {φ(k_d(i+d,d_i)) | 1 ≤ i ≤ ⌊(t-l)/d⌋}, where l is the block length, d is the sliding stride between adjacent blocks, and φ is a learnable MLP with intra-block position encoding to map keys in a block to a single compressed key. k_t^cmp ∈ R^|k|×⌈t/l⌉ is tensor composed by compression keys. Usually, we adopt d < l to mitigate information fragmentation. An analogous formulation holds for the compressed value representation v_t^cmp. Compressed representations capture coarser-grained higher-level semantic information and reduce computational burden of attention."
    },
    {
      "title": "Token Selection",
      "content": "Using only compressed keys, values might lose important fine-grained information, motivating us to selectively preserve individual keys, values. Below we describe our efficient token selection mechanism that identifies and preserves the most relevant tokens with low computational overhead. Blockwise Selection. Our selection strategy processes key and value sequences in spacial continuous blocks, motivated by two key factors: hardware efficiency considerations and inherent distribution patterns of attention scores. Blockwise selection is crucial to achieve efficient computation on modern GPUs. That is because modern GPU architectures exhibit significantly higher throughput for continuous block accesses compared to random index-based reads. Also, blockwise computation enables optimal utilization of Tensor Cores. This architectural characteristic has established blockwise memory access and computation as a fundamental principle in high-performance attention implementations, as exemplified by FlashAttention’s block-based design. Blockwise selection follows the inherent distribution patterns of attention scores. Prior works [Jiang et al.2024] have shown that attention scores often exhibit spatial continuity, suggesting that neighboring keys tend to share similar importance levels. Our visualization in Section[6.2] also shows this spatial continuous pattern. To implement blockwise selection, we first divide key, value sequences into selection blocks. To identify the most important blocks for attention computation, we need to assign importance scores to each block. Below we present our method for computing these block-level importance scores. Importance Score Computation. Computing block importance scores could introduce significant overhead. Fortunately, the attention computation of compression tokens produces intermediate attention scores that we can leverage to induce selection block importance scores, formulated as: p_t^cmp = Softmax(q_t, k_t^cmp), where p_t^cmp ∈ R^⌈t/l⌉ is the attention scores between q_t and compression keys k_t^cmp. Let l' denote the selection block size. When compression blocks and selection blocks share the same blocking scheme, i.e., l' = l = d, we can directly obtain the selection block importance scores p_t^slc by p_t^slc = p_t^cmp. For cases where the blocking schemes differ, we derive the importance scores for selection blocks according to their spatial relationship. Given d | l and d | l', we have: p_t^slc[j] = ∑_{m=0}^{⌊(l'-1)/d⌋} ∑_{n=0}^{⌊(l'-1)/d⌋} p_t^cmp[m+d*j+n], where | denotes the indexing operator for accessing vector element. For models employing GQA or MQA where key-value caches are shared across query heads, consistent block selection across these heads has to be ensured to minimize KV cache loading during decoding. The shared importance scores across heads in a group are formally defined as: p_t^slc^h = ∑_{h=1}^H p_t^slc(h), where (h) in the superscript denotes the head index, and H is the number of query heads in each group. This aggregation ensures consistent block selection across heads within the same group."
    },
    {
      "title": "Top-n Block Selection",
      "content": "After obtaining the selection block importance scores, We retain tokens within the top-n sparse blocks ranked by block importance scores, formulated as: I_t = {i | rank(p_t^slc[i]) ≤ n}, k_t^slc = Cat{[k_t^slc[i:(i+1)∖l] | i ∈ I_t]}, where rank(.) denotes the ranking position in descending order, with rank = 1 corresponding to the highest score, I_t is the set of selected blocks’ indices, Cat denotes the concatenation operation, k_t^slc ∈ R^|k|×n⌈t/l⌉ is tensor composed by compression keys. An analogous formulation applies to the fine-grained value v_t^slc. The selected keys and values then participate in the attention computation with q_t as defined in Equation (3)."
    },
    {
      "title": "Sliding Window",
      "content": "In attention mechanisms, local patterns typically adapt faster and can dominate the learning process, potentially preventing the model from effectively learning from compression and selection tokens. To address this issue, we introduce a dedicated sliding window branch that explicitly handles local context, allowing other branches (compression and selection) to focus on learning their respective features without being short-circuited by local patterns. Specifically, we maintain recent tokens k_t^win = k_t^win, v_t^win = v_t^win in a window w_t, and isolate attention computations of different information sources (compression tokens, and selected tokens, sliding window) into separate branches. These branch outputs are then aggregated through a learned gating mechanism. To further prevent shortcut learning across attention branches with marginal computational overhead, we provide independent keys and values for three branches. This architectural design enables stable learning by preventing gradient interference between local and long-range pattern recognition, while introducing minimal overhead. After obtaining all three categories of keys and values (k_t^cmp, v_t^cmp ; k_t^slc, v_t^slc; and k_t^win, v_t^win), we compute the final attention output following Equation (5). Together with the compression, selection, and sliding window mechanisms described above, this forms the complete algorithmic framework of NSA."
    },
    {
      "title": "Kernel Design",
      "content": "To achieve FlashAttention-level speedup during the training and prefiling, we implement hardware-aligned sparse attention kernels upon Triton. Given MHA is memory-intensive and inefficient for decoding, we focus on architectures with shared KV caches like GQA and MQA following the current sate-of-the-art LLMs. While compression and sliding window attention computations are readily compatible with existing FlashAttention-2 kernels, we introduce the specialized kernel design for sparse selection attention. If we were to follow FlashAttention’s strategy of leading temporally continuous query blocks into SRAM, it would result in inefficient memory access since queries within a block may require disjoint KV blocks. To address this, our key optimization lies in a different query grouping strategy: for each position on the query sequence, we load all query heads within a GQA group (they share the same sparse KV blocks) into SRAM. Figure[2] illustrates our forward pass implementation. The proposed kernel architecture is characterized by the following key features: 1. Group-Centric Data Loading. For each inner loop, load all heads’ queries Q ∈ R^|h×d| in the group at position t and their shared sparse key/value block indices I_t. 2. Shared KV Fetching. In the inner loop, Sequentially load continuous key/value blocks indexed by I_t into SRAM as K ∈ R^|k|×d_l, v ∈ R^|v|×d_l to minimize memory loading, where b_k is the kernel block size satisfying b_k|l'. 3. Outer Loop on Grid. Since the inner-loop length (proportional to the selected block count n) remains nearly identical for different query blocks, we put query/output loops in Triton’s grid scheduler to simplify and optimize the kernel. This design achieves near-optimal arithmetic intensity by (1) eliminating redundant KV transfers through group-wise sharing, and (2) balancing compute workloads across GPU streaming multiprocessors."
    }
  ],
  "figures": [
    {
      "title": "Figure 1",
      "description": "Kernel design for NSA. The kernel loads queries by GQA groups (Grid Loop), fetches corresponding sparse KV blocks (Inner Loop), and performs attention computation on SRAM. Green blocks indicate data on SRAM, while blue indicates data on HBMI."
    },
    {
      "title": "Figure 2",
      "description": "Pretraining loss comparison between Full Attention and our NSA on 27B-parameter model. Both models exhibit stable convergence, with NSA achieving lower loss values."
    },
    {
      "title": "Figure 3",
      "description": "Training loss curve showing the comparison between Full Attention and NSA during pretraining."
    }
  ]
}
```

Processing pages 11 to 15...
Chunk summary:
```json
{
  "title": "Performance comparison between our NSA and baselines on LongBench",
  "authors": "NSA outperformed most of the baselines including Full Attention",
  "abstract": "We evaluated the pretrained NSA and Full Attention baseline on various benchmarks including MMLU, MMLU-FRO, CMMLU, BBH, GSM8K, MATH, DROP, MBPP, and HumanEval. NSA achieved superior overall performance, outperforming all baselines including Full Attention on 7 out of 9 metrics.",
  "sections": [
    {
      "title": "General Evaluation",
      "content": "Despite its sparsity, NSA achieves superior overall performance, outperforming all baselines including Full Attention on 7 out of 9 metrics."
    },
    {
      "title": "Long-Context Evaluation",
      "content": "NSA achieves perfect retrieval accuracy across all positions in 64k-context needle-in-a-haystack test. This performance stems from our hierarchical sparse attention design."
    },
    {
      "title": "Chain-of-Thought Reasoning Evaluation",
      "content": "NSA demonstrates exceptional performance on tasks requiring complex reasoning over long contexts, achieving +0.087 and +0.051 improvements over Full Attention on multi-hop QA tasks."
    },
    {
      "title": "Efficiency Analysis",
      "content": "We evaluated the computational efficiency of NSA against Full Attention on an 8-GPU A100 system. NSA achieves progressively greater speedups as context length increases, up to 9.0x forward and 6.0x backward speedup at 64k context-length."
    }
  ],
  "figures": [
    {
      "title": "Needle-in-a-Haystack retrieval accuracy",
      "description": "NSA achieves perfect accuracy through its hierarchical sparse attention design."
    },
    {
      "title": "Training loss curve",
      "description": "Compare training loss on a 3B-parameter model with Full Attention and different token selection strategies. Our NSA achieves better performance."
    }
  ]
}
```

Processing pages 16 to 20...
Chunk summary:
```json
{
  "title": "Related Works",
  "authors": [],
  "abstract": "We review existing approaches that improve the efficiency of attention computation through sparse attention. These methods can be broadly categorized into three groups based on their core strategies: (1) Fixed sparse pattern, (2) Dynamic token pruning, and (3) query-aware selection. We introduce several representative works from each category.",
  "sections": [
    {
      "title": "Fixed Sparse Pattern",
      "content": "SlidingWindow is a commonly used approach that allows the query to compute attention only within a fixed window. StreamingLLM addresses the challenges of processing long text streams by maintaining two critical portions of the context: an attention sink (early tokens) and a local context window. While these approaches effectively reduce memory and computation costs, their rigid pattern of ignoring contexts limits their performance on tasks requiring full context understanding."
    },
    {
      "title": "Dynamic Token Pruning",
      "content": "H2O implements an adaptive approach to reduce KV-cache memory usage during decoding. This method dynamically evicts tokens deemed less important for future predictions based on their recent utility according to attention score. SnapKV also introduces a token pruning strategy that reduces the KV cache by selectively retaining only the most crucial features, enabling efficient memory usage. SnapKV identifies important features through attention weight analysis and voting during preilling, then updates KV cache by combining selected compressed features with recent context to maintain prompt consistency."
    },
    {
      "title": "Query-Aware Selection",
      "content": "Quest employs a blockwise selection strategy where each chunk’s importance is estimated by product between query and coordinate-wise min-max of the key chunks. The results scores help to select top-n important key-value chunks for attention. InfLLM combines fixed patterns with retrieval by maintaining attention sinks, local context, and retrievable chunks. This method selects representative keys from each chunk to estimate chunk importance. FlashAttention formulates pivotal token identification as a recommendation problem by mapping queries and keys to Hamming space using learned functions. ClusterKV achieves sparsity by firstly clustering keys and then selecting the most relevant clusters for attention computation based on query-cluster similarity."
    }
  ],
  "figures": []
}
```

Processing pages 21 to 24...
Chunk summary:
```json
{
  "title": "Logarithmic and Exponential Equations Solution",
  "authors": [],
  "abstract": "The document presents solutions to systems of logarithmic and exponential equations. It begins with a baseline result where m = 25 and n = 8, giving m + n = 33. It then proceeds to solve a system of logarithmic equations using matrix inversion and Cramer’s rule, resulting in specific values for x, y, and z. The document then solves a problem involving real numbers x and y with x, y > 1, satisfying log3(y^4) = log5(x^4) = 10, and finds the value of xy to be 25.",
  "sections": [
    {
      "title": "Baseline Result",
      "content": "Given m = 25 and n = 8, m + n = 33."
    },
    {
      "title": "System of Logarithmic Equations",
      "content": [
        "Logarithmic equations: log2(x/yz) = 1/2, log2(y/xz) = 1/3, log2(z/xy) = 1/4",
        "Exponential conversions: x = sqrt(2) * yz, y = 2^(1/3) * xz, z = 2^(1/4) * xy",
        "Solving step-by-step: x = sqrt(2) * yz, y = 2^(1/3) * xz, z = 2^(1/4) * xy",
        "Matrix inversion: A = log2(x), B = log2(y), C = log2(z)",
        "System of linear equations: A - B - C = 1/3, B - A - C = 1/4, C - A - B = 1/4",
        "Solution: x = 2^(61/48), y = 2^(13/16), z = 2^(-1/6)",
        "Calculating |log2(x^4 * y^2 * z^2)|: 4 * log2(x) + 3 * log2(y) + 2 * log2(z) = 4 * 61/48 + 3 * 13/16 + 2 * (-1/6) = 131."
      ]
    },
    {
      "title": "Logarithmic Equations with Real Numbers x and y",
      "content": [
        "Given log3(y^4) = 10 and log5(x^4) = 10, find xy.",
        "Using logarithm power rule: log3(y^4) = x * log3(y) = 10 => y = 3^(5/2)",
        "Using power rule: log5(x^4) = y * log5(x) = 10 => 4 * log5(x) = 10 => log5(x) = 5/4 => x = 5^(4/5)",
        "Substituting values: xy = (5^(4/5)) * (3^(5/2)) = 25."
      ]
    }
  ],
  "figures": []
}
```

Aggregating chunk summaries into a final JSON summary...

Final aggregated JSON summary:
```json
{
  "title": "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention",
  "authors": [
    "Jingyang Yuan",
    "Huazhu Gao",
    "Damai Dai",
    "Jureya Luo",
    "Liang Zhao",
    "Zhengyan Zhang",
    "Zhenda Xie",
    "T. X. Wei",
    "Lean Wang",
    "Zhiping Xiao",
    "Yuqiang Wang",
    "Chong Ruan",
    "Ming Zhang",
    "Wenfeng Liang",
    "Wangding Zeng"
  ],
  "abstract": "Long-context modeling is crucial for next-generation language models, yet the high computational cost of standard attention mechanisms poses significant challenges. Sparse attention offers a promising direction for improving efficiency while maintaining model capabilities. We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision. Our approach advances sparse attention design with two key innovations: (1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design, with implementation optimizations for modern hardware. (2) We enable end-to-end training, reducing pretraining computation without sacrificing model performance. As shown in Figure[1] experiments show the model pretrained with NSA maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning. Meanwhile, NSA achieves substantial speedups over Full Attention on 64k-length sequences across decoding, forward propagation, and backward propagation, validating its efficiency throughout the model lifecycle.",
  "sections": [
    {
      "title": "Introduction",
      "content": "The research community increasingly recognizes long-context modeling as a crucial capability for next-generation large language models, driven by diverse real-world applications ranging from in-depth reasoning (DeepSeek-AI [2025], Zelikman et al. [2022]), repository-level code generation (Zhang et al. [2023a], Zhang et al.), and multi-turn autonomous agent systems (Park et al. [2023]). Recent breakthroughs, including OpenAI’s o-series models, DeepSeek-RI (DeepSeek-AI [2023]), and Gemini 1.5 Pro (Google et al. [2024]), enabling models to process entire codebases, lengthy documents, maintain coherent multi-turn conversations over thousands of tokens, and perform complex reasoning across long-range dependencies. However, the high complexity (Zheng et al. [2023]) of vanilla Attention (Vaswani et al. [2017]) mechanisms emerges as a critical latency bottleneck as sequence length increases. Theoretical estimates indicate that attention computation with softmax architectures accounts for 70–80% of total latency when decoding 64k-length contexts, underscoring the urgent need for more efficient attention mechanisms."
    },
    {
      "title": "Overall Framework",
      "content": "To leverage the potential of attention with natural sparse pattern, we propose replacing the original key-value pairs k_q, v_q in Equation (1) with a more compact and information-dense set of representation key-value pairs k_t, v_t, given each query q_t. Specifically, we formally define the optimized attention output as follows: k_t = f_k(q_t, k_q, v_q), v_t = f_v(q_t, k_q, v_q). o_t^q = Attn(q_t, k_t, v_t), where k_t, v_t are dynamically constructed based on the current query q_t and the contextual memory k_q, v_q. We can design various mapping strategies to get different categories of k_t, v_t and combine them as follows: o_t^q = ∑_c∈C g_c^q * Attn(q_t, k_t^c, v_t^c). As illustrated in Figure[2], NSA have three mapping strategies C = {cmp, slc, win}, representing compression, selection, and sliding window for keys and values. g_c ∈ [0,1] is the gate score for corresponding strategy c, derived from input features via an MLP and sigmoid activation. Let N_t denote the total number of remapped keys/values: N_t = ∑_c∈C size[k_t^c]. We maintain a high sparsity ratio by ensuring N_t << t."
    },
    {
      "title": "Algorithm Design",
      "content": "In this subsection, we introduce the design of our remapping strategies f_k and f_v: token compression, token selection, and sliding window. 3.3.1. Token Compression By aggregating sequential blocks of keys or values into block-level representations, we obtain compressed keys and values that capture the information of the entire block. Formally, the compressed key representation is defined as: k_t^cmp = f_k^cmp(k_q) = {φ(k_d(i+d,d_i)) | 1 ≤ i ≤ ⌊(t-l)/d⌋}, where l is the block length, d is the sliding stride between adjacent blocks, and φ is a learnable MLP with intra-block position encoding to map keys in a block to a single compressed key. k_t^cmp ∈ R^|k|×⌈t/l⌉ is tensor composed by compression keys. Usually, we adopt d < l to mitigate information fragmentation. An analogous formulation holds for the compressed value representation v_t^cmp. Compressed representations capture coarser-grained higher-level semantic information and reduce computational burden of attention."
    },
    {
      "title": "Token Selection",
      "content": "Using only compressed keys, values might lose important fine-grained information, motivating us to selectively preserve individual keys, values. Below we describe our efficient token selection mechanism that identifies and preserves the most relevant tokens with low computational overhead. Blockwise Selection. Our selection strategy processes key and value sequences in spacial continuous blocks, motivated by two key factors: hardware efficiency considerations and inherent distribution patterns of attention scores. Blockwise selection is crucial to achieve efficient computation on modern GPUs. That is because modern GPU architectures exhibit significantly higher throughput for continuous block accesses compared to random index-based reads. Also, blockwise computation enables optimal utilization of Tensor Cores. This architectural characteristic has established blockwise memory access and computation as a fundamental principle in high-performance attention implementations, as exemplified by FlashAttention’s block-based design. Blockwise selection follows the inherent distribution patterns of attention scores. Prior works [Jiang et al.2024] have shown that attention scores often exhibit spatial continuity, suggesting that neighboring keys tend to share similar importance levels. Our visualization in Section[6.2] also shows this spatial continuous pattern. To implement blockwise selection, we first divide key, value sequences into selection blocks. To identify the most important blocks for attention computation, we need to assign importance scores to each block. Below we present our method for computing these block-level importance scores. Importance Score Computation. Computing block importance scores could introduce significant overhead. Fortunately, the attention computation of compression tokens produces intermediate attention scores that we can leverage to induce selection block importance scores, formulated as: p_t^cmp = Softmax(q_t, k_t^cmp), where p_t^cmp ∈ R^⌈t/l⌉ is the attention scores between q_t and compression keys k_t^cmp. Let l' denote the selection block size. When compression blocks and selection blocks share the same blocking scheme, i.e., l' = l = d, we can directly obtain the selection block importance scores p_t^slc by p_t^slc = p_t^cmp. For cases where the blocking schemes differ, we derive the importance scores for selection blocks according to their spatial relationship. Given d | l and d | l', we have: p_t^slc[j] = ∑_{m=0}^{⌊(l'-1)/d⌋} ∑_{n=0}^{⌊(l'-1)/d⌋} p_t^cmp[m+d*j+n], where | denotes the indexing operator for accessing vector element. For models employing GQA or MQA where key-value caches are shared across query heads, consistent block selection across these heads has to be ensured to minimize KV cache loading during decoding. The shared importance scores across heads in a group are formally defined as: p_t^slc^h = ∑_{h=1}^H p_t^slc(h), where (h) in the superscript denotes the head index, and H is the number of query heads in each group. This aggregation ensures consistent block selection across heads within the same group."
    },
    {
      "title": "Top-n Block Selection",
      "content": "After obtaining the selection block importance scores, We retain tokens within the top-n sparse blocks ranked by block importance scores, formulated as: I_t = {i | rank(p_t^slc[i]) ≤ n}, k_t^slc = Cat{[k_t^slc[i:(i+1)∖l] | i ∈ I_t]}, where rank(.) denotes the ranking position in descending order, with rank = 1 corresponding to the highest score, I_t is the set of selected blocks’ indices, Cat denotes the concatenation operation, k_t^slc ∈ R^|k|×n⌈t/l⌉ is tensor composed by compression keys. An analogous formulation applies to the fine-grained value v_t^slc. The selected keys and values then participate in the attention computation with q_t as defined in Equation (3)."
    },
    {
      "title": "Sliding Window",
      "content": "In attention mechanisms, local patterns typically adapt faster and can dominate the learning process, potentially preventing the model from effectively learning from compression and selection tokens. To address this issue, we introduce a dedicated sliding window branch that explicitly handles local context, allowing other branches (compression and selection) to focus on learning their respective features without being short-circuited by local patterns. Specifically, we maintain recent tokens k_t^win = k_t^win, v_t^win = v_t^win in a window w_t, and isolate attention computations of different information sources (compression tokens, and selected tokens, sliding window) into separate branches. These branch outputs are then aggregated through a learned gating mechanism. To further prevent shortcut learning across attention branches with marginal computational overhead, we provide independent keys and values for three branches. This architectural design enables stable learning by preventing gradient interference between local and long-range pattern recognition, while introducing minimal overhead. After obtaining all three categories of keys and values (k_t^cmp, v_t^cmp ; k_t^slc, v_t^slc; and k_t^win, v_t^win), we compute the final attention output following Equation (5). Together with the compression, selection, and sliding window mechanisms described above, this forms the complete algorithmic framework of NSA."
    },
    {
      "title": "Kernel Design",
      "content": "To achieve FlashAttention-level speedup during the training and prefiling, we implement hardware-aligned sparse attention kernels upon Triton. Given MHA is memory-intensive and inefficient for decoding, we focus on architectures with shared KV caches like GQA and MQA following the current sate-of-the-art LLMs. While compression and sliding window attention computations are readily compatible with existing FlashAttention-2 kernels, we introduce the specialized kernel design for sparse selection attention. If we were to follow FlashAttention’s strategy of leading temporally continuous query blocks into SRAM, it would result in inefficient memory access since queries within a block may require disjoint KV blocks. To address this, our key optimization lies in a different query grouping strategy: for each position on the query sequence, we load all query heads within a GQA group (they share the same sparse KV blocks) into SRAM. Figure[2] illustrates our forward pass implementation. The proposed kernel architecture is characterized by the following key features: 1. Group-Centric Data Loading. For each inner loop, load all heads’ queries Q ∈ R^|h×d| in the group at position t and their shared sparse key/value block indices I_t. 2. Shared KV Fetching. In the inner loop, Sequentially load continuous key/value blocks indexed by I_t into SRAM as K ∈ R^|k|×d_l, v ∈ R^|v|×d_l to minimize memory loading, where b_k is the kernel block size satisfying b_k|l'. 3. Outer Loop on Grid. Since the inner-loop length (proportional to the selected block count n) remains nearly identical for different query blocks, we put query/output loops in Triton’s grid scheduler to simplify and optimize the kernel. This design achieves near-optimal arithmetic intensity by (1) eliminating redundant KV transfers through group-wise sharing, and (2) balancing compute workloads across GPU streaming multiprocessors."
    },
    {
      "title": "General Evaluation",
      "content": "Despite its sparsity, NSA achieves superior overall performance, outperforming all baselines including Full Attention on 7 out of 9 metrics."
    },
    {
      "title": "Long-Context Evaluation",
      "content": "NSA achieves perfect retrieval accuracy across all positions in 64k-context needle-in-a-haystack test. This performance stems from our hierarchical sparse attention design."
    },
    {
      "title": "Chain-of-Thought Reasoning Evaluation",
      "content": "NSA demonstrates exceptional performance on tasks requiring complex reasoning over long contexts, achieving +0.087 and +0.051 improvements over Full Attention on multi-hop QA tasks."
    },
    {
      "title": "Efficiency Analysis",
      "content": "We evaluated the computational efficiency of NSA against Full Attention on an 8-GPU A100 system. NSA achieves progressively greater speedups as context length increases, up to 9.0x forward and 6.0x backward speedup at 64k context-length."
    },
    {
      "title": "Related Works",
      "content": "We review existing approaches that improve the efficiency of attention computation through sparse attention. These methods can be broadly categorized into three groups based on their core strategies: (1) Fixed sparse pattern, (2) Dynamic token pruning, and (3) query-aware selection. We introduce several representative works from each category."
    },
    {
      "title": "Fixed Sparse Pattern",
      "content": "SlidingWindow is a commonly used approach that allows the query to compute attention only within a fixed window. StreamingLLM addresses the challenges of processing long text streams by maintaining two critical portions of the context: an attention sink (early tokens) and a local context window. While these approaches effectively reduce memory and computation costs, their rigid pattern of ignoring contexts limits their performance on tasks requiring full context understanding."
    },
    {
      "title": "Dynamic Token Pruning",
      "content": "H2O implements an adaptive approach to reduce KV-cache memory usage during decoding. This method dynamically evicts tokens deemed less important for future predictions based on their recent utility according to attention score. SnapKV also introduces a token pruning strategy that reduces the KV cache by selectively retaining only the most crucial features, enabling efficient memory usage. SnapKV identifies important features through attention weight analysis and voting during preilling, then updates KV cache by combining selected compressed features with recent context to maintain prompt consistency."
    },
    {
      "title": "Query-Aware Selection",
      "content": "Quest employs a blockwise selection strategy where each chunk’s importance is estimated by product between query and coordinate-wise min-max of the key chunks. The results scores help to select top-n important key-value chunks for attention. InfLLM combines fixed patterns with retrieval by maintaining attention sinks, local context, and retrievable chunks. This method selects representative keys from each chunk to estimate chunk importance. FlashAttention formulates pivotal token identification as a recommendation problem by mapping queries and keys to Hamming space using learned functions. ClusterKV achieves sparsity by firstly clustering keys and then selecting the most relevant clusters for attention computation based on query-cluster similarity."
    }
  ],
  "figures": [
    {
      "title": "Figure 1",
      "description": "Comparison of performance and efficiency between Full Attention model and our NSA. Left: Despite being sparse, NSA surpasses Full Attention baseline on average across general benchmarks, long-context tasks, and reasoning evaluation. Right: For 64k-length sequence processing, NSA achieves substantial computational speedup compared to Full Attention in all stages: decoding, forward propagation, and backward propagation."
    },
    {
      "title": "Figure 2",
      "description": "Overview of NSA’s architecture. Left: The framework processes input sequences through three parallel attention branches. For a given query, preceding keys and values are processed into compressed attention for coarse-grained patterns, selected attention for important token blocks, and sliding attention for local context. Right: Visualization of different attention patterns produced by each branch. Green areas indicate regions where attention scores need to be computed, while white areas represent regions that can be skipped."
    },
    {
      "title": "Figure 3",
      "description": "Kernel design for NSA. The kernel loads queries by GQA groups (Grid Loop), fetches corresponding sparse KV blocks (Inner Loop), and performs attention computation on SRAM. Green blocks indicate data on SRAM, while blue indicates data on HBMI."
    },
    {
      "title": "Figure 4",
      "description": "Pretraining loss comparison between Full Attention and our NSA on 27B-parameter model. Both models exhibit stable convergence, with NSA achieving lower loss values."
    },
    {
      "title": "Figure 5",
      "description": "Training loss curve showing the comparison between Full Attention and NSA during pretraining."
    },
    {
      "title": "Figure 6",
      "description": "Needle-in-a-Haystack retrieval accuracy. NSA achieves perfect accuracy through its hierarchical sparse attention design."
    },
    {
      "title": "Figure 7",
      "description": "Training loss curve. Compare training loss on a 3B-parameter model with Full Attention and different token selection strategies. Our NSA achieves better performance."
    }
  ],
  "conclusion": "Overall, NSA demonstrates significant improvements in efficiency and performance over traditional attention mechanisms. Its innovative approach to sparse attention, combined with hardware-aligned optimizations, makes it a promising direction for future research in long-context modeling. Experiments validate that NSA not only maintains or exceeds the performance of Full Attention models but also achieves substantial speedups in various computational stages, demonstrating its practicality and effectiveness in real-world applications."
}
```
