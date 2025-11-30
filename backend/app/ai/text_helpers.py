import bisect
import copy

from typing import Any, Dict, List
from underthesea import sent_tokenize, pos_tag
from sentence_transformers import SentenceTransformer

class TextHelpers:
    @staticmethod
    def get_words(text: str) -> List[Dict[str, Any]]:
        with open("/home/nhan/Workspace/All-Courses/Y4-S1/CT552/Application/backend/ai_models/vietnamese-stopwords.txt", "r") as f:
            sorted_stopwords = sorted([w.strip() for w in f.read().split("\n")])
        
        sentences = sent_tokenize(text)
        TAG = {"N", "Np", "V", "A"}
        words = []

        wid = 0
        for sid, sent in enumerate(sentences):
            tokens = pos_tag(sent)
            
            for w, pos_tag_val in tokens: 
                if not any(c.isalnum() for c in w):  
                    continue
                           
                pos_check = bisect.bisect_left(sorted_stopwords, w.lower())
                if pos_check < len(sorted_stopwords) and sorted_stopwords[pos_check] == w.lower():
                    continue
                
                words.append({
                    "wid": wid,
                    "word": w,
                    "pos": pos_tag_val, 
                    "sent_id": sid,
                    "sentence": sent,
                })
                wid += 1
    
    @staticmethod
    def compute_origin_embeddings(words: List[Dict[str, Any]], model: SentenceTransformer) -> List[Dict[str, Any]]:
        temp = copy.deepcopy(words)
        all_words = [w["word"] for w in temp]
        all_sents = [w["sentence"] for w in temp]

        word_embs = model.encode(all_words, convert_to_tensor=True, show_progress_bar=False)
        sent_embs = model.encode(all_sents, convert_to_tensor=True, show_progress_bar=False)

        for i, w in enumerate(temp):
            w["origin_word_embedding"] = word_embs[i].cpu()
            w["origin_sent_embedding"] = sent_embs[i].cpu()

        return temp