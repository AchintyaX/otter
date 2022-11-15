from enum import unique
import pandas as pd 
from sentence_transformers import util, SentenceTransformer
import numpy as np 
from typing import Optional
from loguru import logger


class Community_detection:
    def __init__(self, embedder_model: str, stopword_list: Optional[str] = None) -> None:
        logger.info(f"Loading {embedder_model} for embeddings")
        self.embedder = SentenceTransformer(embedder_model)
        self.stopwords = stopword_list
    
    def remove_stopwords(self, txt):
        txt = txt.lower()
        new_txt = txt.split()
        new_arr = [i for i in new_txt if i not in self.stopwords]
        new_txt = ' '.join(new_arr)
        return new_txt
    

    def fit(self, df: pd.DataFrame,text_column: str, min_community_size: int=25, threshold: float=0.65):
        copy_df = df.copy(deep=True)
        unique_topics = copy_df[text_column].str.lower().unique().tolist()
        if self.stopwords is None:
            unique_topics = list(set(unique_topics))
        else:
            unique_topics = [self.remove_stopwords(i) for i in unique_topics]
            unique_topics = list(set(unique_topics))
        
        corpus_topics = unique_topics
        logger.info("Generating Embedddings")
        corpus_embeddings = self.embedder.encode(unique_topics, batch_size=128, show_progress_bar=True, convert_to_tensor=True)
        logger.info("Running Community detection on Embeddings")
        clusters = util.community_detection(corpus_embeddings, min_community_size=min_community_size, threshold=threshold)

        clustered_sentences = []
        for i, cluster in enumerate(clusters):
            cluster_list = []
            for sentence_id in cluster:
                cluster_list.append(unique_topics[sentence_id])
            clustered_sentences.append(
                {
                    'cluster_id': int(i+1),
                    'cluster_words': cluster_list, 
                    'size': len(cluster)
                }
            )

        cluster_df = pd.DataFrame(clustered_sentences)
        return cluster_df
        