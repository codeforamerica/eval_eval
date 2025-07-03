# SE EVAL^2
SE Eval^2 (Eval Eval) is a Solutions Engineering and Data Science Discovery Cycle project that aims to assess LLM evaluation libraries and services. Here we will implement and compare metrics for a shared AI problem space. The chosen problem space is Supplemental Nutrition Assistance Program (SNAP) notice quality analysis. 

Given ten sample SNAP notices this repository produces sample analysis with model and prompt iteration. The analysis functions as an input to evaluate. Creating domain knowledge of LLM-evaluation frameworks and comparing implementation factors is the goal of the project. Creating a SNAP notice analysis tool is not.

## How to Contribute
To contribute to the Discovery Cycle, choose an evaluation and/or telemetry framework and create a script in the "experiments" folder. If you'd like, [a template script](experiments/template.py) has been provided for convenience.
The `main.py` script should automatically pick it up and run it. After writing and running the evaluation, don't forget to fill out the post evaluation survey.

## Getting Started
To leverage your machine's GPU, we've opted to run this experiment with a Python local environment. We'll need to install Ollama and Python dependencies locally.

1. Install Ollama with homebrew: `brew install ollama`
2. Pull any Ollama models ollama pull "model name". Analysis is currently using "llama3.1:8b" and "deepseek-r1:8b".
3. Setup a Python virtual environment: `python3 -m venv .venv && chmod +x .venv/bin/activate && .venv/bin/activate`
4. Install requirements `.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .`
5. Copy example.env to .env and copy any API keys down from our shared LastPass record.

NB: Steps 1 and 2 are only required if you'd like to run the analysis command or write evaluation with Ollama models.

## Commands
This repository provides two commands "analyze" and "evaluate". The former creates analysis for use in experiments. The latter runs the evaluation experiments and is most relevant for most discovery cycle participation cases.

Sample Execution of "evaluate" command:

Run all experiments:

```shell
.venv/bin/python main.py evaluate manifest_with_analysis.json --output-path="results.json"
```

Run a single experiment:
```shell
.venv/bin/python main.py evaluate manifest_with_analysis.json --output-path="results.json" --metrics=rouge_experiment,another_experiment
```

Run analysis:
```shell
.venv/bin/python main.py analysis manifest.json --output-path="results.json"
```
NB: Running the analysis command is not required for contributing evaluations. Translations are available on [Google Drive](https://drive.google.com/drive/folders/1Ytaw5QCVPBu2KzkCdTRkts00i2m1WE2P?usp=drive_link).

TODO: Update link

## Adding Dependencies
You will almost certainly need to add additional Python packages to contribute evaluations. This repository uses pip and pip-chill to manage dependencies. To add a new dependency run `.venv/bin/pip install <package>`. When you are ready to commit your work, add your dependencies to our requirements file (requirements.txt) by running `.venv/bin/pip-chill > requirements.txt`. Pip will try to add our local package `se-eval-eval==0.0.0` to the requirements. Please remove this dependency from the requirements file, as it will break our installation process.

## Adding Providers and Models
Feel free to add Python packages necessary to support other LLM Providers. See "Adding Dependencies" above. Often, this is handled by the evaluation framework. If you need to add support for a provider that has not been used yet in the project, make sure to include the API key in your .env file, the example.env file and shared LastPass record.

If you need to download or install an additional non-Ollama model, please do so in your experiment file.

If you need to add an Ollama model, include it in the list of supported models in main.py.