# LlaMA-Gomuku

This repo is an emxperimental code for a Large Multi-Modal Model that combine board game and natural language together. The goal is to let alpha-zero-like agent to explain it's strategy in natrual language. 

## Plan
Some main steps:
- Train a projection layer that map policy net features to embedding space of llm
- Bootstrap a instruction fine tuning dataset with inference and reasoning over combined modal
- instruction fine tuning to align with human preference