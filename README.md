# EVAL EVAL
EVAL EVAL (Eval^2) is a Solutions Engineering and Data Science Discovery Cycle project that aims to assess LLM evaluation libraries and services. Here we will implement and compare metrics for a shared AI problem space. The chosen problem space is Supplemental Nutrition Assistance Program (SNAP) notice quality analysis. 

Given ten sample SNAP notices this repository produces sample analysis with model and prompt iteration. The analysis functions as an input to evaluate. Creating domain knowledge of LLM-evaluation frameworks and comparing implementation factors is the goal of the project. Improving the SNAP notice analysis is only an objective when it enables evaluation.

The command line tools in this repository add information to a JSON manifest over analysis and evaluation steps. The manifest file's components are [represented by Pydantic models](eval_eval/schema.py), which allows for easy validation and use with LLMs. Contributors should add evaluations to the [experiments directory](experiments) as described below.

## How to Contribute
To contribute to the Discovery Cycle, choose an evaluation and/or telemetry framework and create a script in the "experiments" folder. If you'd like, [a template script](experiments/template.py) has been provided for convenience.
The `main.py` script should automatically pick it up and run it. After writing and running the evaluation, don't forget to fill out the post evaluation survey.

## Getting Started
To leverage your machine's GPU, we've opted to run this experiment with a Python local environment. We'll need to install Ollama and Python dependencies locally.

You must have python 3.12 or later installed to run the project. Check your python version with `python3 --version`. Newer versions of python can be installed from [python.org](https://python.org).

1. Install Ollama with homebrew: `brew install ollama`
2. Start ollama: `ollama serve`
3. Pull any Ollama models: `ollama pull "<model name>"`. Analysis is currently using "llama3.1:8b" and "deepseek-r1:8b".
4. Setup a Python virtual environment: `python3 -m venv .venv && chmod +x .venv/bin/activate && .venv/bin/activate`
5. Install requirements `pip install -e . && pip install -r requirements.txt`
6. Copy example.env to .env and copy any API keys down from our shared LastPass record.

NB: Steps 1-3 are only required if you'd like to run the analysis command or write evaluation with Ollama models.

## Commands
This repository provides two commands "analyze" and "evaluate". The former creates analysis for use in experiments. The latter runs the evaluation experiments and is most relevant for most discovery cycle participation cases.

Sample Execution of "evaluate" command:

Run all experiments:

```shell
python main.py evaluate manifest_with_analysis.json --output-path="results.json"
```

Run a single experiment:
```shell
python main.py evaluate manifest_with_analysis.json --output-path="results.json" --metrics=rouge_experiment,another_experiment
```

Run analysis:
```shell
python main.py analyze manifest.json --output-path="results.json"
```
NB: Running the analysis command is not required for contributing evaluations. Manifests with and without analysis and notice documents are available on [Google Drive](https://drive.google.com/drive/folders/1Ejh-i1ZrF96tY2HBcuOXHsXussracltp?usp=drive_link).

## Adding Dependencies
You will almost certainly need to add additional Python packages to contribute evaluations. This repository uses pip and pip-chill to manage dependencies. To add a new dependency run `pip install <package>`. When you are ready to commit your work, add your dependencies to our requirements file (requirements.txt) by running `pip-chill > requirements.txt`.

## Adding Providers and Models
Feel free to add Python packages necessary to support other LLM Providers. See "Adding Dependencies" above. Often, this is handled by the evaluation framework. If you need to add support for a provider that has not been used yet in the project, make sure to include the API key in your `.env` file and `example.env` file. Entries in the .env file will be included as environment variables automatically (access with `os.getenv("API_KEY")`).

If you need to download or install an additional non-Ollama model, please do so in your experiment file.

If you need to add an Ollama model, include it in the list of supported models in main.py.
