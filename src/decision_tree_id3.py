import pandas as pd
from utils import entropy, information_gain

# -----------------------------------------------------------
# Build ID3 Decision Tree (recursive)
# -----------------------------------------------------------
def id3(df, target, attributes):
    # Base case: if all labels are same
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # If no attributes left
    if len(attributes) == 0:
        return df[target].mode()[0]  # majority class

    # Choose best attribute (highest information gain)
    gains = {attr: information_gain(df, attr, target) for attr in attributes}
    best_attr = max(gains, key=gains.get)

    tree = {best_attr: {}}

    # Build subtree for each value of best_attr
    for val in df[best_attr].unique():
        subset = df[df[best_attr] == val]

        if subset.empty:
            tree[best_attr][val] = df[target].mode()[0]
        else:
            remaining = [a for a in attributes if a != best_attr]
            tree[best_attr][val] = id3(subset, target, remaining)

    return tree

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("Loan-eligibility-ML\data\cleaned_loan_data.csv")
    attributes = list(df.columns[:-1])
    target = df.columns[-1]

    tree = id3(df, target, attributes)
    print("\nDecision Tree (ID3):\n")
    print(tree)
