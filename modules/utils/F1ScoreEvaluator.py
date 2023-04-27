import faiss
import numpy as np
from tqdm import tqdm


class F1ScoreEvaluator(object):
    """
    Class that evaluates F1 score on given dataset (with labels)
    and embeddings.
    """

    def __init__(self, df, embeddings, k=100):
        self.k = k
        self.df = df
        self.embeddings = embeddings
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.similarities, self.indexes = self.index.search(embeddings, self.k)

    def get_avg_f1_score_for_threshold(self, threshold):
        f1_score_accumulated = 0
        for i in range(len(self.embeddings)):
            cur_sims = self.similarities[i]
            cur_indexes = self.indexes[i]
            duplicate_indexes = cur_indexes[cur_sims >= threshold]
            results = self.df.iloc[duplicate_indexes]['posting_id'].values
            targets = self.df.iloc[i]['target']
            f1_score = self.calc_f1_score(targets, results)
            f1_score_accumulated += f1_score
        return f1_score_accumulated / len(self.embeddings)

    def get_avg_f1_scores_for_thresholds(self, thresholds):
        f1_avg_scores = []
        for threshold in tqdm(thresholds):
            f1_avg = self.get_avg_f1_score_for_threshold(threshold)
            f1_avg_scores.append(f1_avg)

        return f1_avg_scores

    def calc_f1_score(self, targets, results):
        intersect = len(np.intersect1d(targets, results))
        return 2 * intersect / (len(targets) + len(results))
