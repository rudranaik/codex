# Codex Apriori Demo

This repository includes a simple example of generating item recommendations using the Apriori algorithm.

The `apriori_demo.py` script contains:

- A small set of sample transactions.
- Functions to one-hot encode the dataset and generate association rules.
- A CLI that prints recommended items for the provided basket of items.

## Requirements

Install the required dependency:

```bash
pip install pandas mlxtend
```

## Usage

Run the script from the command line, passing the items you currently have:

```bash
python apriori_demo.py bread beer
```

The script will output recommended items based on the learned associations.
