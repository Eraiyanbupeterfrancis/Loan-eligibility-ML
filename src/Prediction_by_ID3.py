import pandas as pd
from decision_tree_id3 import id3
from utils import load_data

# -----------------------------------------------------------
# Recursive function to predict using the tree
# -----------------------------------------------------------
def predict(tree, sample):
    # If the tree is a label, return it ("Yes" / "No")
    if isinstance(tree, str):
        return tree

    # Else it's a dictionary -> {attribute: {value: subtree}}
    attribute = list(tree.keys())[0]
    branches = tree[attribute]

    sample_value = sample.get(attribute)

    # If the value exists in the branches, follow it
    if sample_value in branches:
        return predict(branches[sample_value], sample)
    else:
        # Unknown value -> fallback to majority "No"
        return "No"


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    # Load training data
    df = load_data("Loan-eligibility-ML/data/cleaned_loan_data.csv")

    attributes = list(df.columns[:-1])
    target = df.columns[-1]

    # Train decision tree
    tree = id3(df, target, attributes)

    print("\nTrained Decision Tree:\n", tree)

    # Get user input
    print("\nEnter Applicant Information:")
    income = input("Income (Low / Medium / High): ").strip().capitalize()
    credit = input("CreditScore (Poor / Fair / Good): ").strip().capitalize()
    employment = input("Employment (Yes / No): ").strip().capitalize()
    age = input("Age (Young / Adult / Senior): ").strip().capitalize()

    # Prepare input as dictionary for prediction
    sample = {
        "Income": income,
        "CreditScore": credit,
        "Employment": employment,
        "Age": age
    }

    # Make prediction
    result = predict(tree, sample)

    print("\nPrediction:")
    if result == "Yes":
        print("✅ Loan Approved (Eligible)")
    else:
        print("❌ Loan Not Approved (Not Eligible)")
