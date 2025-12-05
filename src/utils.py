import pandas as pd
import math
from collections import Counter

# -----------------------------------------------------------
# Load dataset safely
# -----------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    df = df.applymap(lambda x: str(x).strip())    # clean spaces
    return df

# -----------------------------------------------------------
# Entropy calculation
# -----------------------------------------------------------
def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0

    counts = Counter(labels)

    ent = 0
    for count in counts.values():
        p = count / total
        ent += -p * math.log2(p)

    return ent

# -----------------------------------------------------------
# Information Gain
# -----------------------------------------------------------
def information_gain(df, attribute, target):
    total_entropy = entropy(df[target])
    values = df[attribute].unique()

    weighted_entropy = 0

    for val in values:
        subset = df[df[attribute] == val]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy
