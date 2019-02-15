# -*- coding: utf-8 -*-
"""
scikit-xtend Machine Learning Library Extensions
Created on 2019

@author: Panagiotis Katsaroumpas and David S. Batista <dsbatista@gmail.com>
http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV

class EstimatorSelectionHelper:
    """
    from sklearn import datasets
    
    breast_cancer = datasets.load_breast_cancer()
    X_cancer = breast_cancer.data
    y_cancer = breast_cancer.target
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.svm import SVC
    
    models1 = {
        'ExtraTreesClassifier': ExtraTreesClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'SVC': SVC()
    }
    
    params1 = {
        'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
        'RandomForestClassifier': { 'n_estimators': [16, 32] },
        'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
        'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
        'SVC': [
            {'kernel': ['linear'], 'C': [1, 10]},
            {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
        ]
    }
    """
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.grid_searches[key] = gs    

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]