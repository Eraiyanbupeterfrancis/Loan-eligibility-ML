import pandas as pd
import math
from collections import Counter

# -----------------------------------------------------------
# Load dataset
# -----------------------------------------------------------
def load_data(path):
    return pd.read_csv(path)

# -----------------------------------------------------------
# Entropy calculation
# -----------------------------------------------------------
def entropy(labels):
    total = len(labels)
    counts = Counter(labels)

    return -sum((count/total) * math.log2(count/total) for count in counts.values())

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
