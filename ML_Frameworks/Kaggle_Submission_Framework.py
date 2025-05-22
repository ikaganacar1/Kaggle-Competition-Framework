import pandas as pd


class Submission:
    def __init__(self, preds_df, ids, submission_name="submission.csv"):
        df = pd.DataFrame(
            {
                "id": ids,
                "preds": preds_df,
            }
        )
        df.to_csv(submission_name, index=False)

        print("Submission file saved successfully!")
        print(df.head())
