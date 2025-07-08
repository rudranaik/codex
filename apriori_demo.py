import sys
from typing import List

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction data
TRANSACTIONS = [
    ['bread', 'milk'],
    ['bread', 'diaper', 'beer', 'egg'],
    ['milk', 'diaper', 'beer', 'coke'],
    ['bread', 'milk', 'diaper', 'beer'],
    ['bread', 'milk', 'diaper', 'coke'],
]


def create_dataset(transactions: List[List[str]]) -> pd.DataFrame:
    """One-hot encode the transaction data."""
    all_items = sorted({item for trans in transactions for item in trans})
    encoded_records = []
    for trans in transactions:
        record = {item: (item in trans) for item in all_items}
        encoded_records.append(record)
    return pd.DataFrame(encoded_records)


def generate_rules(df: pd.DataFrame, min_support=0.2, min_confidence=0.6) -> pd.DataFrame:
    """Generate association rules using the Apriori algorithm."""
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    return rules


def recommend(items: List[str], rules: pd.DataFrame, top_n: int = 3) -> List[str]:
    """Recommend items based on the provided list using association rules."""
    recommendations = (
        rules[rules['antecedents'].apply(lambda x: set(items).issuperset(x))]
        .sort_values('confidence', ascending=False)
    )
    suggested = []
    for _, row in recommendations.iterrows():
        consequents = row['consequents']
        for item in consequents:
            if item not in items and item not in suggested:
                suggested.append(item)
            if len(suggested) >= top_n:
                return suggested
    return suggested


def main(args: List[str]):
    df = create_dataset(TRANSACTIONS)
    rules = generate_rules(df)
    if not args:
        print("No items provided. Example usage: python apriori_demo.py bread beer")
        return
    recs = recommend(args, rules)
    if recs:
        print("Recommended items:", ', '.join(recs))
    else:
        print("No recommendations found for provided items.")


if __name__ == "__main__":
    main(sys.argv[1:])
