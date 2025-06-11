# SE EVAL^2
SE Eval^2 (Eval Eval) is a Solutions Engineering Discovery Cycle project that aims to assess LLM evaluation libraries and services. Here we will implement and compare metrics for a shared AI problem space. The chosen problem space is LLM text translation.
Given ten sample texts, we will use LLMs (Ollama) to translate into four common languages, Spanish, Chinese, Vietnamese and Tagalog. These languages were chosen for being highly spoken in California and Washington, where our sample documents come from.

_NB: This repository is a work in progress._

## How to Contribute
To contribute to the Discovery Cycle, choose an evaluation and/or telemetry framework and create a script in the "experiments" folder. If you'd like, [a template script](experiments/template.py) has been provided for convenience.
The `main.py` script should automatically pick it up and run it. After writing and running the evaluation, don't forget to fill out the post evaluation survey.

## Getting Started
To most easily leverage your machine's GPU, we've opted to run this experiment with a Python local environment. We'll need to install Ollama and Python dependencies locally.

1. Install Ollama with homebrew: `brew install ollama`
2. Setup a Python virtual environment: `python3 -m venv .venv && chmod +x .venv/bin/activate && .venv/bin/activate`
3. Install requirements `.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`
4. Pull any Ollama models ollama pull "model name"

## Commands
This repository provides two commands "preprocess" and "evaluate". The former creates initial translation assets required for running experiments. The latter runs the evaluation experiments and is most relevant for most discovery cycle participation cases.

Sample Execution of "evaluate" command:

Run all experiments:

```
.venv/bin/python main.py evaluate translations.json --output-path="results.json"
```

Run a single experiment (coming soon):
```
.venv/bin/python main.py evaluate translations.json --output-path="results.json" --experiment-name="rouge_experiment"
```