import pandas as pd
from utils import entropy, information_gain

# -----------------------------------------------------------
# Build ID3 Decision Tree (recursive)
# -----------------------------------------------------------
def id3(df, target, attributes):
    
    # Base case 1: all labels identical
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # Base case 2: no attributes left
    if len(attributes) == 0:
        return df[target].mode()[0]   # majority class

    # Compute information gain for each attribute
    gains = {attr: information_gain(df, attr, target) for attr in attributes}
    
    # Select the best attribute
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}

    # Build each branch of the tree
    for val in df[best_attr].unique():
        subset = df[df[best_attr] == val]

        if subset.empty:
            tree[best_attr][val] = df[target].mode()[0]
        else:
            remaining = [a for a in attributes if a != best_attr]
            tree[best_attr][val] = id3(subset, target, remaining)

    return tree


# -----------------------------------------------------------
# Pretty Print Decision Tree
# -----------------------------------------------------------
def print_tree(tree, indent=0):
    """Prints the tree nicely."""
    if not isinstance(tree, dict):
        print(" " * indent + "→ " + tree)
        return
    
    for attr, branches in tree.items():
        print(" " * indent + f"[ATTRIBUTE: {attr}]")
        for value, subtree in branches.items():
            print(" " * (indent + 2) + f"({value}) →")
            print_tree(subtree, indent + 4)


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    # ✔ Correct Windows path
    df = pd.read_csv("Loan-eligibility-ML/data/cleaned_loan_data.csv")

    attributes = list(df.columns[:-1])
    target = df.columns[-1]

    print("\nBuilding ID3 Decision Tree...\n")
    tree = id3(df, target, attributes)

    print("\nFinal Decision Tree:\n")
    print_tree(tree)
