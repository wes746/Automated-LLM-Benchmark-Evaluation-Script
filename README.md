# Automated LLM Benchmark Evaluation

## Overview
This repository provides a streamlined Python script to objectively evaluate the performance of fine-tuned Large Language Models (LLMs) and LoRA adapters. 

It is built on top of EleutherAI's `lm-evaluation-harness`. By testing models against established academic benchmarks, developers can mathematically verify whether a fine-tuning run improved or degraded the model's reasoning capabilities.

## Default Benchmarks
The script is pre-configured to run three standard benchmarks:
1. **HellaSwag:** Tests common-sense natural language inference.
2. **ARC Challenge:** Tests grade-school science questions requiring complex logic.
3. **TruthfulQA:** Measures the model's propensity to generate falsehoods or hallucinations.

## Requirements and Installation
This script requires a CUDA-enabled GPU. Install the necessary dependencies via your terminal:

```bash
pip install lm-eval[hf]
