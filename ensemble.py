
import pandas as pd
from os import listdir
from os.path import isfile, join

mypath = "./pcam/submissions_for_ensemble"

csv_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

print(csv_files)

csv_df_list = [pd.read_csv(file, header=0) for file in csv_files]



cases = csv_df_list[0].loc[:,"case"]

predictions_list = []
for df in csv_df_list:
    predictions = df.loc[:,"prediction"]
    predictions_list.append(predictions)

df = pd.DataFrame()

for index, pred in enumerate(predictions_list):
    df[f"prediction_{index}"] = pred


ensemble_df = pd.DataFrame()
ensemble_df["case"] = cases



def majority_vote(all_preds):
    votes = []
    for value in all_preds:
        if value > 0.50:
            votes.append(1)
        else:
            votes.append(0)

    return sum(votes)/len(votes)


def majority_vote_soft(all_preds):
    return sum(all_preds)/len(all_preds)

maj_votes = []
for index, row in df.iterrows():
    all_preds = []
    for value in row:
        all_preds.append(value)
    maj_vote = majority_vote_soft(all_preds)
    maj_votes.append(maj_vote)
    
ensemble_df["prediction"] = maj_votes

print(ensemble_df)

ensemble_df.to_csv('./pcam/ensemble_submission_TTA_soft_BESTONE.csv',index=False)