import os
from typing_extensions import Self
from urllib import response
import pandas as pd
import numpy as np
import re
from sentence_transformers import util, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import torch
import pandas as pd
from tqdm import tqdm
import copy
from typing import Any, Tuple, List, Dict
from loguru import logger
from otter.metrics import intersection_score, get_classification_report


class semantic_matching:
    """
    Weak Supervised Classification on Text Data using bi-encoder based ranking followed by cross-encoder based reranking
    """

    def __init__(
        self, taxonomy_dict, embedder_model: str, cross_encoder_model: str
    ) -> None:
        logger.info(f"loading {embedder_model} embedder ")
        self.embedder = SentenceTransformer(embedder_model)

        logger.info(f"loading {cross_encoder_model} cross encoder")
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        self.taxonomy_dict = taxonomy_dict
        self.taxonomy_arr = [val for key, val in taxonomy_dict.items()]
        self.taxonomy_keys = list(taxonomy_dict.keys())
        logger.info("encoding the labels")
        self.taxonomy_embeddings = self.embedder.encode(
            self.taxonomy_arr, convert_to_numpy=True, show_progress_bar=True
        )

    def text_cleaning(self, text: str) -> str:
        CLEANR = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        text = re.sub(CLEANR, "", text)
        text = re.sub(r"http\S+", " ", text)
        text = text.replace("\xa0", " ")
        text = text.replace("\n", " ")
        # text = " ".join(re.sub("@([a-zA-Z0-9_ ]{1,50})","", text).split())
        return text

    def search_rank(self, query: str, rank_list_size: int = 5) -> Any:
        taxonomy_embeddings = copy.deepcopy(self.taxonomy_embeddings)
        taxonomy_keys = copy.deepcopy(self.taxonomy_keys)
        taxonomy_dict = copy.deepcopy(self.taxonomy_dict)

        if len(query) == 0:
            logger.info("Empty string passed")
            return "others", None, []

        query_vec = self.embedder.encode(query)
        cos_scores = util.cos_sim(query_vec, taxonomy_embeddings)

        top_results = torch.topk(cos_scores, k=rank_list_size)
        top_results_indices = top_results.indices.tolist()[0]

        potential_labels = []
        potential_label_names = []
        for i in top_results_indices:
            potential_labels.append(taxonomy_dict[taxonomy_keys[i]])
            potential_label_names.append(taxonomy_keys[i])

        comp_list = [[query, i] for i in potential_labels]

        cross_encoder_results = self.cross_encoder.predict(comp_list)
        score_positions = np.argpartition(cross_encoder_results, -3)[-3:]
        labels_metadata = []
        for i in score_positions:
            score = cross_encoder_results[i]
            temp_dict = {"label_name": potential_label_names[i], "score": str(score)}
            labels_metadata.append(temp_dict)

        score = np.amax(cross_encoder_results)
        result = potential_label_names[int(np.where(cross_encoder_results == score)[0])]

        return result, str(score), labels_metadata

    def predict(self, query: str) -> Dict[Any, Any]:
        query = self.text_cleaning(query)

        top_label, top_label_score, labels_metadata = self.search_rank(query=query)
        response_dict = {
            "top_label": top_label,
            "top_label_score": top_label_score,
            "labels_metadata": labels_metadata,
        }

        return response_dict

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

        copy_df["semantic_label"] = top_labels
        copy_df["semantic_score"] = top_label_scores
        copy_df["semantic_metadata"] = labels_metadata

        return copy_df

    def intersection_score(self, df: pd.DataFrame, label_column: str, text_column: str):

        output_df = self.predict_df(df=df, text_column=text_column)
        score = intersection_score(
            df=output_df, prediction_column="semantic_label", target_column=label_column
        )
        return score

    def classification_report(
        self, df: pd.DataFrame, label_column: str, text_column: str
    ):

        output_df = self.predict_df(df=df, text_column=text_column)
        report = get_classification_report(
            df=output_df, prediction_column="semantic_label", target_column=label_column
        )

        return report
