import pandas as pd

# -----------------------------------------------------------
# Helper: Check if h1 is more general than h2
# -----------------------------------------------------------
def is_more_general(h1, h2):
    return all(h1[i] == "?" or h1[i] == h2[i] for i in range(len(h1)))


# -----------------------------------------------------------
# Helper: Generate minimal specializations of G for negatives
# -----------------------------------------------------------
def min_specializations(h, example, attribute_values):
    specializations = []

    for i in range(len(h)):
        if h[i] == "?":   # only "?" can be specialized
            for val in attribute_values[i]:
                if val != example[i]:  # avoid matching negative example
                    new_h = h.copy()
                    new_h[i] = val
                    specializations.append(new_h)

    return specializations


# -----------------------------------------------------------
# MAIN Candidate Elimination Function
# -----------------------------------------------------------
def candidate_elimination(path):

    df = pd.read_csv(path)

    # Extract domain values for each attribute
    attributes = df.columns[:-1]
    label_col = df.columns[-1]

    attribute_values = [df[col].unique().tolist() for col in attributes]

    num_attr = len(attributes)

    # Initialize S (most specific) and G (most general)
    S = [["Ø"] * num_attr]
    G = [["?"] * num_attr]

    # -------------------------------------------------------
    # Process each training example
    # -------------------------------------------------------
    for _, row in df.iterrows():

        example = row.iloc[:-1].tolist()
        label = row.iloc[-1]

        # ===============================================
        #          POSITIVE EXAMPLE
        # ===============================================
        if label == "Yes":

            # 1. Remove inconsistent G
            G = [g for g in G if is_more_general(g, example)]

            # 2. Generalize S minimally
            new_S = []
            for s in S:
                new_h = s.copy()

                for i in range(num_attr):
                    if s[i] == "Ø":
                        new_h[i] = example[i]
                    elif s[i] != example[i]:
                        new_h[i] = "?"

                new_S.append(new_h)
            S = new_S

            # 3. Remove overly general S
            S = [s for s in S if any(is_more_general(g, s) for g in G)]

        # ===============================================
        #          NEGATIVE EXAMPLE
        # ===============================================
        else:

            # 1. Remove S that incorrectly classify negative examples
            S = [s for s in S if not is_more_general(s, example)]

            # 2. Specialize G minimally
            new_G = []
            for g in G:
                if is_more_general(g, example):  # g wrongly covers negative
                    specs = min_specializations(g, example, attribute_values)

                    # keep only consistent specializations
                    for spec in specs:
                        if any(is_more_general(g2, spec) for g2 in G) and \
                           any(is_more_general(spec, s) for s in S):
                            new_G.append(spec)
                else:
                    new_G.append(g)

            G = new_G

    return S, G


# -----------------------------------------------------------
# Run the algorithm using your CLEANED CSV
# -----------------------------------------------------------
if __name__ == "__main__":
    S_final, G_final = candidate_elimination("Loan-eligibility-ML\data\cleaned_loan_data.csv")

    print("\nFinal Specific Boundary (S):")
    for s in S_final:
        print("  ", s)

    print("\nFinal General Boundary (G):")
    for g in G_final:
        print("  ", g)
