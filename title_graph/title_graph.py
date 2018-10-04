import pickle
import re
import string

import pkg_resources

from gensim.models import KeyedVectors
import numpy as np


class Preprocessor(object):
    char_search = re.compile(r"[^\u0020\u0027\u002b-\u002e\u0030-\u0039\u0041-\u005a\u0061-\u007a]")
    strip_multi_ws = re.compile(r"( {2,})")
    word_re = re.compile(r"([\w|-]+)")
    punc = set(string.punctuation)

    def __init__(self):
        self.kp = self._load_kp()
        self.gram_counts = self._load_gram_counts()

    def __call__(self, x):
        x = self.char_search.sub(" ", x)
        x = self.strip_multi_ws.sub(" ", x)
        x = self.word_re.findall(x)
        x = [w.lower() for w in x if len(w) > 1 and w not in self.punc]
        # trimmed_x = self._trim_title(" ".join(x))
        return " ".join(x)

    def _load_kp(self):
        data_path = pkg_resources.resource_filename('title_graph', 'data/kp.pkl')
        with open(data_path, "rb") as pfile:
            return pickle.load(pfile)

    def _load_gram_counts(self):
        data_path = pkg_resources.resource_filename('title_graph', 'data/gram_counts.pkl')
        with open(data_path, "rb") as pfile:
            return pickle.load(pfile)

    def _trim_title(self, x):
        matches = self.kp.extract_keywords(x)
        if not matches:
            return x
        if len(matches) > 1:
            return max([(kw, self.gram_counts.get(kw, 0)) for kw in matches], key=lambda x: x[1])[0]
        else:
            return matches[0]


class TitleGraph(object):

    def __init__(self, preprocessor=Preprocessor):
        self.graph = self._load_graph()
        self.model = self._load_model()
        self.preprocessor = preprocessor()

    def _load_graph(self):

        data_path = pkg_resources.resource_filename('title_graph', 'data/graph.pkl')
        with open(data_path, "rb") as pfile:
            return pickle.load(pfile)

    def _load_model(self):

        data_path = pkg_resources.resource_filename('title_graph', 'data/title_model.kv')
        return KeyedVectors.load(data_path, mmap='r')

    def query_forward(self, title, min_weight=2, topn=25):

        """

        Given a Job Title, find the most likely Job Title to occur next

        :param title: str, a Job Title
        :param min_weight: int, the minimum weight to consider from the graph. Setting this higher will reduce the number of matches returned
        :return: results if title in self.graph else None
        """

        x = self.preprocessor(title)
        if x not in self.graph:
            return None

        results = [(x, y) for x, y in self.graph.succ.get(x).items()]

        result_vecs = []
        for title, data in results:
            if data['weight'] < min_weight:
                continue
            td = {'title': title, 'weight': data['weight'], 'vec': self.model.wv.get_vector(title) * data['weight']}
            result_vecs.append(td)

        if not result_vecs:
            return []

        resulting_vec = np.mean([x['vec'] for x in result_vecs], axis=0)
        return self.model.wv.similar_by_vector(resulting_vec, topn=topn)

    def query_backwards(self, title, min_weight=2, topn=25):

        """

        Given a Job Title, find the most likely previous Job Title

        :param title: str, a Job Title
        :param min_weight:
        :param topn: int, The number of results to return
        :return: results if title in self.graph else None
        """

        x = self.preprocessor(title)
        if x not in self.graph:
            return None

        results = [(x, y) for x, y in self.graph.pred.get(x).items()]

        result_vecs = []
        for title, data in results:
            if data['weight'] < min_weight:
                continue
            td = {'title': title, 'weight': data['weight'], 'vec': self.model.wv.get_vector(title) * data['weight']}
            result_vecs.append(td)

        if not result_vecs:
            return []

        resulting_vec = np.mean([x['vec'] for x in result_vecs], axis=0)
        return self.model.wv.similar_by_vector(resulting_vec, topn=topn)

    def query_similar_semantic(self, title, topn=25, as_tokens=False):

        """

        Given a Job Title, use FastText via Gensim and return topn similar titles

        :param title: str, a Job Title
        :param topn: int, The number of results to return
        :param as_tokens: bool, Whether to split the string. This should only effect Job Title queries with 2+ words.
            If the order of the words is important, leave as False.
        :return: results
        """

        x = self.preprocessor(title)
        if as_tokens:
            x = x.split()

        return self.model.most_similar(x, topn=topn)
