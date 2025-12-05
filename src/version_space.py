from itertools import product

def is_more_general(h1, h2):
    return all(h1[i] == "?" or h1[i] == h2[i] for i in range(len(h1)))

# -----------------------------------------------------------
# Generate FULL Version Space from S and G
# -----------------------------------------------------------
def generate_version_space(S, G, attribute_values):
    version_space = []

    # Generate all possible hypotheses
    all_hypotheses = list(product(*attribute_values))

    for h in all_hypotheses:
        h = list(h)

        # must be >= S
        if not all(is_more_general(h, s) for s in S):
            continue

        # must be <= some G
        if not any(is_more_general(g, h) for g in G):
            continue

        version_space.append(h)

    return version_space


# -----------------------------------------------------------
# Demo (Optional)
# -----------------------------------------------------------
if __name__ == "__main__":
    # Example
    S = [['?', 'Good', 'Yes', '?']]
    G = [['?', 'Good', 'Yes', '?']]
    attribute_values = [
        ['Low', 'Medium', 'High'],
        ['Poor', 'Fair', 'Good'],
        ['Yes', 'No'],
        ['Young', 'Adult', 'Senior']
    ]

    VS = generate_version_space(S, G, attribute_values)
    print("Version Space:")
    for h in VS:
        print(h)
