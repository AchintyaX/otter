import os
from urllib import response 
import pandas as pd 
import numpy as np 
import re
from sentence_transformers import util, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import torch

import copy
from typing import Any, Tuple, List, Dict
from loguru import logger




class semantic_matching:
    def __init__(self, taxonomy_dict, embedder_model: str, cross_encoder_model: str) -> None:
        logger.info(f"loading {embedder_model} embedder ")
        self.embedder = SentenceTransformer(embedder_model)

        logger.info(f"loading {cross_encoder_model} cross encoder")
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        self.taxonomy_dict = taxonomy_dict
        self.taxonomy_arr = [val for key, val in taxonomy_dict.items()]
        self.taxonomy_keys = list(taxonomy_dict.keys())
        logger.info("encoding the labels")
        self.schemma_embeddings = self.embedder.encode(self.taxonomy_arr, convert_to_numpy=True, show_progress_bar=True)
    
    def text_cleaning(self, text: str) -> str:
        CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        text = re.sub(CLEANR, '', text)
        text=re.sub(r'http\S+', ' ', text)
        text=text.replace('\xa0', ' ')
        text=text.replace('\n', ' ')
        #text = " ".join(re.sub("@([a-zA-Z0-9_ ]{1,50})","", text).split())
        return text
    
    def search_rank(self, query: str) -> Any:
        schemma_embeddings = copy.deepcopy(self.schemma_embeddings)
        taxonomy_keys = copy.deepcopy(self.taxonomy_keys)
        taxonomy_dict = copy.deepcopy(self.taxonomy_dict)

        if len(query) == 0:
            logger.info("Empty string passed")
            return 'others', None, []


        query_vec = self.embedder.encode(query)
        cos_scores = util.cos_sim(query_vec, schemma_embeddings)
        
        top_results = torch.topk(cos_scores, k=5)
        top_results_indices = top_results.indices.tolist()[0]
        
        potential_themes = []
        potential_theme_names = []
        for i in top_results_indices:
            potential_themes.append(taxonomy_dict[taxonomy_keys[i]])
            potential_theme_names.append(taxonomy_keys[i])
        
        comp_list = [[query, i] for i in potential_themes]
        
        cross_encoder_results = self.cross_encoder.predict(comp_list)
        score_positions = np.argpartition(cross_encoder_results, -3)[-3:]
        themes_metadata = []
        for i in score_positions:
            score = cross_encoder_results[i]
            temp_dict = {
                'theme_name': potential_theme_names[i],
                'score':str(score)
            }
            themes_metadata.append(temp_dict)
        
        score = np.amax(cross_encoder_results)
        result = potential_theme_names[int(np.where(cross_encoder_results == score)[0])]
        
        return result, str(score), themes_metadata
    
    def predict(self, query: str) -> Dict[Any, Any]:
        query = self.text_cleaning(query)

        top_theme, top_theme_score, themes_metadata = self.search_rank(query=query)
        response_dict = {
            "top_theme":top_theme, 
            "top_theme_score": top_theme_score, 
            "themes_metadata": themes_metadata
        }

        return response_dict

        


