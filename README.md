# MoE Experiment - Stream Experts Research

> **Research Project:** Running Large AI Models with Limited Resources using Mixture of Experts

This is an experimental research project exploring **Stream Experts** techniques to run large AI models (Mixture of Experts) on consumer hardware with CPU and limited RAM.

## 🎯 Research Goals

This project investigates how to:
- **Load and run large MoE models** (like Mixtral 8x7B) on CPU-only systems
- **Stream experts dynamically** instead of loading all into memory
- **Use GGUF quantization** to dramatically reduce memory footprint
- **Enable local inference** for models typically requiring expensive GPUs

## 📚 Background: What are Mixture of Experts (MoE)?

**Mixture of Experts** is a machine learning technique where a large model is composed of multiple specialized sub-models (experts). During inference, only a subset of experts is activated for each input, enabling:

- **Massive model capacity** (billions of parameters) with manageable compute
- **Sparse activation**: Only ~10-20% of parameters are used per token
- **Efficient pretraining** with conditional computation

### Key Insight for CPU Inference

Traditional MoE models like Mixtral 8x7B have 47B parameters but use only 2 experts per token (out of 8). This means:
- **Active computation**: ~12B parameters per forward pass
- **Memory challenge**: All 47B parameters must be accessible

**Our research focuses on:** Streaming experts from disk on-demand, enabling CPU inference without loading the full model into RAM.

## 🔬 Research Approaches

### 1. GGUF Format Support
- **GGUF** (GPT-Generated Unified Format) allows efficient storage of quantized models
- Supports various quantization levels (Q4_K_M, Q5_K_M, Q8_0)
- Enables running 70B+ models with <8GB RAM

### 2. Dynamic Expert Loading
Instead of loading all 8 experts:
- Load router/gating network (small)
- Load shared attention layers
- **Stream experts from disk** as needed based on routing decisions
- Cache recently-used experts in RAM

### 3. Memory Mapping (mmap)
- Use operating system's virtual memory to map model weights
- Load only accessed portions into RAM
- Let OS handle paging to/from disk

## 🚀 Current Implementation

```python
from moe_experiment import load_model

# Load a quantized MoE model with memory mapping
model = load_model(
    model_path="mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    repo_id="TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
    use_mmap=True,  # Enable memory mapping
    lazy_load=True,  # Stream experts on-demand
)

# Run inference
output = model.generate("Explain quantum computing in simple terms")
```

## 📦 Installation

### Requirements
- Python 3.10+
- 8GB+ RAM (16GB recommended)
- ~50GB free disk space for models

### Setup

```bash
# Clone and enter the repository
git clone https://github.com/arthur999999/moe-experiment.git
cd moe-experiment

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

## 🔍 Research Areas

### Active Investigations

1. **Expert Caching Strategies**
   - LRU cache for frequently-used experts
   - Preloading experts based on context
   - Compression of expert weights

2. **Quantization Impact**
   - Compare Q4 vs Q5 vs Q8 quantization
   - Measure quality degradation vs memory savings
   - Optimal quantization per expert type

3. **Streaming Overhead**
   - Disk I/O latency for expert loading
   - Prefetching strategies
   - NVMe vs SSD vs HDD performance

4. **Batch Processing**
   - Efficient routing of multiple tokens
   - Parallel expert loading
   - Throughput optimization

## 📊 Benchmarks

| Model | Format | Total Params | RAM Required | Inference Speed |
|-------|--------|--------------|--------------|-----------------|
| Mixtral 8x7B | FP16 | 47B | ~94 GB | N/A (GPU only) |
| Mixtral 8x7B | Q4_K_M | 47B | ~6 GB | ~5 tokens/sec* |
| Mixtral 8x7B | Q4_K_M + Stream | 47B | ~4 GB | ~2 tokens/sec* |

*Approximate speeds on CPU (AMD Ryzen 7 5800X)

## 🛠️ Development

```bash
# Run tests
pytest -v

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Run benchmarks
python benchmarks/throughput.py --model mixtral-8x7b --quantization Q4_K_M
```

## 📚 Resources

### Papers
- [Adaptive Mixture of Local Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) - Original MoE paper
- [Switch Transformers (2022)](https://arxiv.org/abs/2101.03961) - Scaling to trillion parameters
- [Mixture of Experts Explained](https://huggingface.co/blog/moe) - Hugging Face overview

### Models
- [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) - State-of-the-art open MoE
- [GGUF Quantized Models](https://huggingface.co/TheBloke) - Community quantized versions

### Related Projects
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF inference in C/C++
- [gguf-py](https://github.com/ggerganov/llama.cpp/tree/master/gguf-py) - GGUF format handling

## 🤝 Contributing

This is a research project. Contributions welcome!

- Report issues with specific models/hardware
- Share benchmark results
- Propose caching strategies
- Implement streaming optimizations

## 📝 Research Notes

### Why GGUF?

GGUF format is specifically designed for:
- Efficient storage of large models
- Multiple quantization schemes
- Metadata for model configuration
- Memory-mapped file support

### Challenges

1. **Expert Switching Latency**: Loading experts from disk adds latency
2. **Context Length**: Long contexts require more memory
3. **Router Accuracy**: Poor routing wastes computation

### Future Directions

- [ ] Expert merging/compression
- [ ] Speculative expert loading
- [ ] Hierarchical expert organization
- [ ] Quantization-aware training

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Hugging Face Hub for model hosting
- TheBloke for quantized GGUF models
- llama.cpp community for GGUF format
- Mistral AI for Mixtral architecture

---

**Note**: This is experimental research software. Performance and memory usage will vary based on hardware and model configuration.
