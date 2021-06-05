import torch
import numpy as np

class Searcher():
    def __init__(self, embs, paths):
                    
        self.embs = embs
        self.paths = paths
        self.pid = [p.split('/')[-1].split('.')[0] for p in paths]
        self.query_chunk_size = 100
        self.emb_chunk_size = 500000
        
    
    def search(self, query: torch.Tensor, top_k: [int]):

        if isinstance(query, (np.ndarray, np.generic)):
            query = torch.from_numpy(query)
        elif isinstance(query, list):
            query = torch.stack(query)

        if len(query.shape) == 1:
            query = query.unsqueeze(0)

        #Check that corpus and queries are on the same device
        if query.device != self.embs.device:
            query = query.to(self.embs.device)

        queries_result_list = [[] for _ in range(len(query))]
        max_top_k = max(top_k)
        
        for query_start_idx in range(0, len(query), self.query_chunk_size):
            # Iterate over chunks of the corpus
            for emb_start_idx in range(0, len(self.embs), self.emb_chunk_size):
                # Compute cosine similarites
                cos_scores = self.cos_sim(
                    query[query_start_idx:query_start_idx+self.query_chunk_size], 
                    self.embs[emb_start_idx:emb_start_idx+self.emb_chunk_size]
                )

                # Get top-k scores
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(max_top_k, len(cos_scores[0])), dim=1, largest=True, sorted=False)
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(cos_scores)):
                    for sub_emb_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                        emb_id = emb_start_idx + sub_emb_id
                        query_id = query_start_idx + query_itr 
                        
                        queries_result_list[query_id].append({
                                'path': self.paths[emb_id],
                                'pid': self.pid[emb_id], 
                                'score': score
                        })

        #Sort and strip to top_k results
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
            queries_result_list[idx] = queries_result_list[idx][0:top_k[idx]]
            
        return queries_result_list


    def cos_sim(self, a: torch.Tensor, b: torch.Tensor):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))