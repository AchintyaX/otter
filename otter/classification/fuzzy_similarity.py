from thefuzz import process
from tqdm import tqdm
import pandas as pd
import re
import numpy as np
from loguru import logger
import copy
from typing import Dict, Any
from otter.metrics import intersection_score, get_classification_report


class fuzzy_classifier:
    def __init__(self, taxonomy_dict) -> None:
        self.taxonomy_dict = taxonomy_dict
        self.inv_map = {v: k for k, v in taxonomy_dict.items()}
        self.taxonomy_arr = [val for key, val in taxonomy_dict.items()]

    def text_cleaning(self, text: str) -> str:
        CLEANR = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        text = re.sub(CLEANR, "", text)
        text = re.sub(r"http\S+", " ", text)
        text = text.replace("\xa0", " ")
        text = text.replace("\n", " ")
        # text = " ".join(re.sub("@([a-zA-Z0-9_ ]{1,50})","", text).split())
        return text

    def predict(self, text):
        inv_map = copy.deepcopy(self.inv_map)
        taxonomy_arr = copy.deepcopy(self.taxonomy_arr)
        text = self.text_cleaning(text=text)
        if len(text) == 0:
            return {
                "top_label": "others",
                "top_label_score": 0,
                "labels_metadata": None,
            }
        result_arr = process.extract(text, taxonomy_arr, limit=3)
        labels_metadata = [
            {"label_name": inv_map[i[0]], "score:": i[1]} for i in result_arr
        ]

        return {
            "top_label": labels_metadata[0]["label_name"],
            "top_label_score": labels_metadata[0]["score"],
            "labels_metadata": labels_metadata,
        }

    def predict_df(self, df: pd.DataFrame, text_column: str):
        copy_df = df.copy(deep=True)
        logger.info(f"getting predictions from {text_column} column ")
        top_labels = []
        top_label_scores = []
        labels_metadata = []
        for text in tqdm(copy_df[text_column]):
            result = self.predict(text)
            top_labels.append(result["top_label"])
            top_label_scores.append(result["top_label_score"])
            labels_metadata.append(result["labels_metadata"])

        copy_df["fuzzy_label"] = top_labels
        copy_df["fuzzy_score"] = top_label_scores
        copy_df["fuzzy_metadata"] = labels_metadata

        return copy_df

    def intersection_score(self, df: pd.DataFrame, label_column: str, text_column: str):

        output_df = self.predict_df(df=df, text_column=text_column)
        score = intersection_score(
            df=output_df, prediction_column="fuzzy_label", target_column=label_column
        )
        return score

    def classification_report(
        self, df: pd.DataFrame, label_column: str, text_column: str
    ):

        output_df = self.predict_df(df=df, text_column=text_column)
        report = get_classification_report(
            df=output_df, prediction_column="fuzzy_label", target_column=label_column
        )

        return report
