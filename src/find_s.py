import pandas as pd
def find_s(datapath):
    df= pd.read_csv(datapath)
    positive_examples=df[df['LoanApproved'] == 'Yes'].iloc[:,:-1].values
    hypothesis=['Ø'] * positive_examples.shape[1]
    for examples in positive_examples:
        for i in range(len(examples)):
            if hypothesis[i] == 'Ø':
                hypothesis[i] = examples[i]
            elif hypothesis[i] != examples[i]:
                hypothesis[i] = '?'
    return hypothesis
if __name__ == "__main__":
    final_hypothesis = find_s('Loan-eligibility-ML\data\loan_dataset_600.csv')
    print("Final Hypothesis: ", final_hypothesis)
