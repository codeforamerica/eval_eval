# SE EVAL^2
SE Eval^2 (Eval Eval) is a Solutions Engineering Discovery Cycle project that aims to assess LLM Evaluation libraries and services. Here we will implement and compare metrics for a shared AI problem space. The chosen problem space is LLM text translation.
Given ten sample texts, we will use LLMs (Ollama) to translate into three common languages, Chinese, Vietnamese and Tagalog. These languages were chosen for being highly spoken in California and Washington, where our sample documents come from.

_NB: This repository is a work in progress._

## How to Contribute
To contribute to the Discovery Cycle, choose and evaluation and/or telemetry framework and implement a EvalExperimentBase class in the experiments directory.
The `main.py` script should automatically pick it up and run it. After writing and running the evaluation, don't forget to fill out the post evaluation survey.

## Getting Started
To most easily leverage your machine's GPU, we've opted to run this experiment with a Python local environment. We'll need to install Ollama and Python dependencies locally.

1. Install Ollama with homebrew: `brew install ollama`
2. Setup a Python virtual environment: `python3 -m venv .venv && chmod +x .venv/bin/activate && .venv/bin/activate`
3. Install requirements `.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`
4. Pull any Ollama models ollama pull "<model>"
