import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import classification_report


def intersection_score(df: pd.DataFrame, prediction_column: str, target_column: str):
    score = 0
    for _, row in df.iterrows():
        if type(row[prediction_column]) == str:
            prediction_set = {row[prediction_column]}
        else:
            prediction_set = set(row[prediction_column])

        target_set = set(row[target_column])

        intersection = target_set.intersection(prediction_set)

        if len(intersection) > 0:
            score += 1

    tot_size = len(df[prediction_column])

    return score / tot_size


def get_classification_report(
    df: pd.DataFrame, prediction_column: str, target_column: str
):
    report = classification_report(df[target_column], df[prediction_column])
    print(report)
    return report
