# SubLayer Streaming Execution (SM-LSE)

**A memory-efficient inference technique for large neural networks on low-spec devices.**

[▶ **Download Full Paper (Google Drive)**](https://drive.google.com/file/d/1nPZtkgNNbHBgs-iJfCIEN3UB3Wtr86II/view?usp=share_link)

## Overview

As AI models continue to scale, running them on memory-constrained devices like mobile phones or low-end laptops becomes almost impossible.  
This project proposes **SubLayer Streaming Execution (SM-LSE)**—a technique that performs inference by **loading only one or two sublayers at a time**, minimizing memory footprint without sacrificing much speed.

## Key Features

- Run 120-layer models on 8GB MacBook Air
- Up to **10× memory savings**
- **2.5× speed-up** using 2-layer sliding window
- No need for model compression, quantization, or compilation

## Architecture

- Model: Deep MLP (60 Linear + 60 ReLU layers)
- Input size: 4096
- Total layers: 120
- Each layer saved separately as `.pth` file
- Structure saved as `structure.pt`

## Inference Strategies

1. **Standard**
   - Load full model into memory
   - Fastest (1.52s), but requires ~4GB RAM

2. **1-Layer SubLayer Streaming**
   - Load one sublayer at a time
   - Very low memory (~1159MB)
   - Slow (21.63s) due to disk I/O

3. **2-Layer Sliding Streaming**
   - Load two sublayers at once (sliding window)
   - Balanced: ~322MB memory, 8.42s runtime
   - **Recommended** for real-world low-RAM inference
  
   - 
