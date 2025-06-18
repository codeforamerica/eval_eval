# SE EVAL^2
SE Eval^2 (Eval Eval) is a Solutions Engineering Discovery Cycle project that aims to assess LLM evaluation libraries and services. Here we will implement and compare metrics for a shared AI problem space. The chosen problem space is LLM text translation.
Given ten sample texts, we will use LLMs (Ollama) to translate into several languages, Spanish, Chinese, Vietnamese, Tagalog, Lao, Panjabi, Russian and Korean. These languages represent a range of high and low resourced translation coverage. 

## How to Contribute
To contribute to the Discovery Cycle, choose an evaluation and/or telemetry framework and create a script in the "experiments" folder. If you'd like, [a template script](experiments/template.py) has been provided for convenience.
The `main.py` script should automatically pick it up and run it. After writing and running the evaluation, don't forget to fill out the post evaluation survey.

## Getting Started
To most easily leverage your machine's GPU, we've opted to run this experiment with a Python local environment. We'll need to install Ollama and Python dependencies locally.

1. Install Ollama with homebrew: `brew install ollama`
2. Setup a Python virtual environment: `python3 -m venv .venv && chmod +x .venv/bin/activate && .venv/bin/activate`
3. Install requirements `.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`
4. Pull any Ollama models ollama pull "model name". Translation is currently using "aya-expanse:8b" and "mistral-nemo:latest".
5. Copy example.env to .env and copy any API keys down from our shared LastPass record.

## Commands
This repository provides two commands "preprocess" and "evaluate". The former creates initial translation assets required for running experiments. The latter runs the evaluation experiments and is most relevant for most discovery cycle participation cases.

Sample Execution of "evaluate" command:

Run all experiments:

```shell
.venv/bin/python main.py evaluate translations.json --output-path="results.json"
```

Run a single experiment:
```shell
.venv/bin/python main.py evaluate translations.json --output-path="results.json" --metrics=rouge_experiment,another_experiment
```

Run translation:
```shell
.venv/bin/python main.py translate manifest.json --output-path="results.json"
```
NB: Running the translation command is not required for contributing evaluations. Translations are available on [Google Drive](https://drive.google.com/drive/folders/1Ytaw5QCVPBu2KzkCdTRkts00i2m1WE2P?usp=drive_link).

## Adding Dependencies
You will almost certainly need to add additional Python packages to contribute evaluations. This repository uses pip and pip-chill to manage dependencies. To add a new dependency run `.venv/bin/pip install <package>`. When you are ready to commit your work, add your dependencies to our requirements file (requirements.txt) by running `.venv/bin/pip-chill > requirements.txt`.

## Adding Providers and Models
Feel free to add Python packages necessary to support other LLM Providers. See "Adding Dependencies" above. Often, this is handled by the evaluation framework. If you need to add support for a provider that has not been used yet in the project, make sure to include the API key in your .env file, the example.env file and shared LastPass record.

If you need to download or install an additional non-Ollama model, please do so in your experiment file.

If you need to add an Ollama model, include it in the list of supported models in main.py.