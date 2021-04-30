from sklearn.metrics import f1_score
import pandas as pd
from sklearn.metrics import classification_report
import pandas as pd
from tqdm.notebook import tqdm
from auto_classifier import utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from auto_classifier.vectorizers import (
    SupervisedVectorizer,
    SupervisedStackedVectorizer,
    StackedVectorizer,
)


class BaseAutoLinearModel:
    def __init__(
        self, estimator, params, X_train, X_test, y_train, y_test, scoring_funcs_dict
    ):
        self.estimator = estimator
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.scoring_func_dict = scoring_funcs_dict

        self.vectorizers = None
        self.report = None

    def vectorizers_prepare(self):
        """Create self.vectorizers -> dict()  {vec_name: vectorizer}"""
        raise NotImplemented

    def model_selection_report(self):
        assert self.vectorizers is not None, "Prepare vectorizers dict. Use vectorizers_prepare()"
        result = {"vectorizer": [], "model": [], "params": []}
        for label in self.y_train.unique():
            result[f"{label}_precision"] = []
            result[f"{label}_recall"] = []
        result.update({k: [] for k in self.scoring_func_dict})
        with tqdm(total=len(self.vectorizers)*len(self.params)) as progress_bar:
            for vec_name in self.vectorizers:
                train_features = self.vectorizers[vec_name].transform(self.X_train)
                test_features = self.vectorizers[vec_name].transform(self.X_test)
                for param in self.params:
                    clasifier = self.estimator()
                    clasifier.set_params(**param)
                    clasifier.fit(train_features, self.y_train)
                    preds = clasifier.predict(test_features)
                    classification_reports = classification_report(
                        self.y_test, preds, output_dict=True
                    )
                    for label in self.y_train.unique():
                        result[f"{label}_precision"].append(
                            classification_reports[str(label)]["precision"]
                        )
                        result[f"{label}_recall"].append(
                            classification_reports[str(label)]["recall"]
                        )
                    for score in self.scoring_func_dict:
                        metric = self.scoring_func_dict[score](self.y_test, preds)
                        result[score].append(metric)
                    result["params"].append(param)
                    result["vectorizer"].append(vec_name)
                    result["model"].append(clasifier)
                    progress_bar.set_description('Model: {}, score: {}'.format(str(clasifier), metric))
                    progress_bar.update()
        self.report = pd.DataFrame(result)
        return self.report


class AutoLinearModel(BaseAutoLinearModel):
    def vectorizers_prepare(self):
        """Create self.vectorizers -> dict() with vectorizers"""
        self.vectorizers = {}
        count_word = CountVectorizer(
            analyzer="word", ngram_range=(1, 1), max_features=15000, max_df=0.7
        )
        count_word.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        count_char = CountVectorizer(
            analyzer="char", ngram_range=(1, 4), max_features=30000, max_df=0.7
        )
        count_char.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        count_ngram = CountVectorizer(
            analyzer="char", ngram_range=(4, 4), max_features=30000, max_df=0.7
        )
        count_ngram.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        tfidf_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1), max_features=15000, max_df=0.7
        )
        tfidf_word.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        tfidf_char = TfidfVectorizer(
            analyzer="char", ngram_range=(1, 4), max_features=30000, max_df=0.7
        )
        tfidf_char.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        tfidf_ngram = TfidfVectorizer(
            analyzer="char", ngram_range=(4, 4), max_features=30000, max_df=0.7
        )
        tfidf_ngram.fit(pd.concat([self.X_train, self.X_test], ignore_index=True))
        supervised_ngram = SupervisedVectorizer(count_ngram, self.y_train)
        supervised_ngram.fit(self.X_train)
        supervised_word = SupervisedVectorizer(count_word, self.y_train)
        supervised_word.fit(self.X_train)

        self.vectorizers["tfidf_ngram_level"] = tfidf_ngram
        self.vectorizers["tfidf_word_level"] = tfidf_word
        self.vectorizers["supervised_ngram_level"] = supervised_ngram
        self.vectorizers["supervised_word_level"] = supervised_word