# Dynamic Gatekeeper for RAG

## Overview
This project implements a Batch-Relative Filtering Algorithm for
Retrieval-Augmented Generation (RAG) systems.

Instead of fixed thresholds, the algorithm dynamically adapts
its strictness based on the score distribution of the current batch.

## Key Features
- Adaptive filtering using batch statistics
- Noise reduction in high-quality batches
- Signal salvaging in low-quality batches
- Guaranteed minimum document retention

## Algorithm
Threshold is computed as:

threshold = mean(scores) + alpha * std(scores)

If too few documents pass, the top-k documents are retained.

## Usage

```bash
python gatekeeper.py
