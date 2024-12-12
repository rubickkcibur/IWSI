# IWSI
This is the official code repo for Paper **Importance Weighting can Help Large Language Models Self-Improve** in AAAI 2025. For the version with full supplementary materials, please refer to `https://arxiv.org/abs/2408.09849`

## Self-Generate samples
To let LLM self-generate samples, use the script `generate_sample_parallel.sh`

## Compute DS weights
To compute DS weights, use the script `compute_weights.sh`

## Train
To start training, use the script `baseline.sh`.

## Evaluation
To evaluate the model, use the script `evaluation.sh`. Please note that if you use your own evaluation script, the accuracy computed may be different from what we report in the paper, because different strategies will parse different answer from the LLM's output text.

**We would recommend that you refer to the arxiv version for more details.**
