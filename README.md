# Generative-Project

# Cost-Optimizing Image-to-Text Generation with BLIP and BLIP-2

**Course:** IE 7615 Deep Learning for AI — Northeastern University  
**Authors:** Sam Yuen, Ragul Magesh, Nithin Raja, Carina Davis

---

## Overview

This project builds an image captioning pipeline using Vision Transformer-based models and large language model decoders, with a focus on cost-effectiveness and interpretability of design decisions. Rather than immediately adopting the most powerful available model, we varied one component at a time — model architecture, fine-tuning strategy, training data volume, and decoding configuration — to isolate the contribution of each choice and identify where compute is actually worth spending.

## Approach

All experiments used the COCO Captions dataset. The project progressed through two model generations, starting with Salesforce BLIP and upgrading to BLIP-2 with Meta's OPT-2.7B decoder. Three fine-tuning strategies were evaluated (full fine-tuning, LoRA, and IA3) alongside four decoding strategies (greedy, beam search, sampling, and top-p). Beam width and temperature were treated as hyperparameters and analyzed empirically.

## Key Findings

- Parameter-efficient fine-tuning (LoRA, IA3) consistently outperformed full fine-tuning when training data was limited — full fine-tuning actively hurt BLIP-2 performance at small data scales
- Beam search with width 5 was the most reliable decoding strategy; wider beams offered no meaningful ROI
- More training time mattered as much as more data — doubling epochs on 3k images outperformed 5 epochs on 10k images in several metrics
- Engineering optimizations (parallel image downloading, bfloat16 precision) reduced epoch time from ~1 hour to a fraction of that with no impact on model quality
- BLIP-2's pre-trained weights are strong enough that 10k images was insufficient to push LoRA beyond the zero-shot baseline, reinforcing that fine-tuning benefit scales with the gap between pre-training domain and target task

## Final Model

**BLIP-2 (blip2-opt-2.7b) + LoRA | 10 epochs | Cosine LR | Beam Search (width 5)**  
CIDEr: **0.915** on a 50-sample COCO held-out evaluation set
