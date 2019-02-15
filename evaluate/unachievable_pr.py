# -*- coding: utf-8 -*-
"""
scikit_xtend Machine Learning Library Extensions
Created on 2019

@author: Alex Xu <ayx2@case.edu>

Based on Unachievable Region in Precision-Recall Space Paper
"""

import numpy as np
from math import log
from sklearn.metrics import average_precision_score, precision_recall_curve, \
    auc

def min_precision(y_true, y_pred):
    """
    Precision (p) and recall (r) must satisfy,

    .. math::
        {p} \\geq \\frac{{s}{r}}{1 - {s} + {s r}}
    where :math:`s` is the skew and :math:`r` is the recall

    We can derive the minimum precision possible with skew of the data
    , given the recall values.

    Theorem 1 from Paper:
        Unachievable Region in Precision-Recall Space
        by Boyd, Costa, Davis, and Page 2012

    Parameters
    ----------
    y_true : array, shape (n_samples,)
        True targets.

    y_pred : array, shape (n_samples,)
        Classications or Probabilities of the positive class.

    Returns
    -------
    min_precision : float (if average is not None) or array of float, , shape =
        [n_unique_labels]

    References
    ----------
    .. [1] `Unachievable Region in Precision-Recall Space
            by Boyd, Costa, Davis, and Page 2012
            <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3858955/>

    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 0, 0, 1])
    >>> min_precision(y_true, y_pred)
    ... # doctest: +ELLIPSIS
    array([0.66666667, 0.6       , 0.        ])
    """
    # Get arrays of preicisions and recalls
    _, r, _ = precision_recall_curve(y_true, y_pred)
    # Calculate skew
    s = np.count_nonzero(y_true)/y_true.size
    # Calculate minimum precision for given recall score and skew of labels
    min_precision = np.array(list(map(lambda r: (s*r)/(1-s+s*r), r)))
    return(min_precision)


def aucpr_min(y_true, recall_bounds=[0, 1]):
    """Minimum AUC for Precision Recall Curve, for given a skew between positive
    and negative labels.

    .. math::
        \\text{AUCPRmin} = \\ {b} - {a} + \\frac{1 -{s}}{{s}} \\ ln(\\frac{
                {s}({a}-1)+1}{{s}({b}-1)+1})
    where :math:`s` is the skew of the data, :math:`a` is the lower recall
    bound, and :math:`b` is upper recall bound

    Corollary 4 and Theorem 2 from Paper:
        Unachievable Region in Precision-Recall Space
        by Boyd, Costa, Davis, and Page 2012

    Parameters
    ----------
    y_true : array, shape = [n_samples,]
        True binary labels or binary label indicators.
        1=Positive Labels, 0=Negative Labels

    recall_bounds : list of int or float, len=2, default=[0, 1]
        Lower and Upper bound of Recalls to calculate AUCPRmin over.
        Default defines whole recall space

    Returns
    -------
    aucpr_min : float

    References
    ----------
    .. [1] `Unachievable Region in Precision-Recall Space
            by Boyd, Costa, Davis, and Page 2012
            <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3858955/>

    Relevance:
    "The existence of the minimum AUCPR and minimum AP can affect the
    qualitative interpretation of a model's performance. For example, changing
    the skew of a data set from 0.01 to 0.5 (e.g., by subsampling the negative
    examples) increases the minimum AUCPR by approximately 0.3. (See Examples)

    "This leads to an automatic jump of 0.3 in AUCPR simply by changing the
    data set and with absolutely no change to the learning algorithm"

    Examples
    --------
    >>> y_true = np.append(np.zeros(99),1)
    >>> aucpr_min(y_true)
    ... # doctest: +ELLIPSIS
    0.005016750503356371
    >>> y_true = np.append(np.zeros(50), np.ones(50))
    >>> aucpr_min(y_true)
    ... # doctest: +ELLIPSIS
    0.3068528194400547
    """
    # Calculate skew
    s = np.count_nonzero(y_true)/y_true.size
    # Get a and b values
    a = recall_bounds[0]
    b = recall_bounds[1]
    # Calculate AUCPRmin
    aucpr_min = b - a + ((1-s)/(s))*log((s*(a-1)+1)/(s*(b-1)+1))
    return(aucpr_min)


def min_average_precision(y_true):
    """The minimum Average Precision Score
    (like sklearn.metrics.average_precision_score)
    , for positive (pos) and negative (neg) examples, respectively, is

    .. math::
        \\text{APmin} = \\frac{1}{pos} \\sum_{i=1}^{pos}\\frac{i}{i + {neg}}

    Theorem 3 from Paper:
        Unachievable Region in Precision-Recall Space
        by Boyd, Costa, Davis, and Page 2012

    Parameters
    ----------
    y_true : array, shape = [n_samples,]
        True binary labels or binary label indicators.
        1=Positive Labels, 0=Negative Labels

    Returns
    -------
    min_average_precision : float

    References
    ----------
    .. [1] `Unachievable Region in Precision-Recall Space
            by Boyd, Costa, Davis, and Page 2012
            <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3858955/>

    Relevance:
    "The existence of the minimum AUCPR and minimum AP can affect the
    qualitative interpretation of a model's performance. For example, changing
    the skew of a data set from 0.01 to 0.5 (e.g., by subsampling the negative
    examples) increases the minimum AUCPR by approximately 0.3. (See Examples)

    "This leads to an automatic jump of 0.3 in AUCPR simply by changing the
    data set and with absolutely no change to the learning algorithm"

    Examples
    --------
    >>> y_true = np.append(np.zeros(99), 1)
    >>> min_average_precision(y_true)
    ... # doctest: +ELLIPSIS
    0.01
    >>> y_true = np.append(np.zeros(50), np.ones(50))
    >>> min_average_precision(y_true)
    ... # doctest: +ELLIPSIS
    0.3118278206898048
    """
    # Number of Positive Labels
    pos = np.count_nonzero(y_true)
    # Number of Negative Labels
    neg = y_true.size - pos
    # Create i to sum over
    i = np.arange(1, pos+1)
    min_average_precision = (1/pos) * np.sum(i/(i+neg))
    return(min_average_precision)


def normalized_aucpr(y_true, y_score, recall_bounds=[0, 1],
                     normalize_strategy='average_precision'):
    """Normalized Area under the Precision-Recall curve. From AUCPR, we
    subtract the minimum AUCPR, so the worst ranking has a score of 0, then
    normalize so the best ranking has a score of 1.

    .. math::
        \\text{AUCNPR} = \\frac{\\text{AUCPR} - \\text{AUCPRmin}}
        {\\text{AUCPRmax} - \\text{AUCPRmin}}
    where :math:`AUCPRmax = 1` when calculating area under entire PR curve
    and :math:`AUCPRmax = b - a` when restricting recall to
    :math:`a\\geq r\\geq b`

    Regardless of skew, the best possible classifier will have an AUCNPR of 1
    and the worst possible classifier will have an AUCNPR of 0. AUCNPR also
    preserves the ordering of algorithms on the same test set since
    AUCPRmax and AUCPRmin are constant for the same data set. Thus, AUCNPR
    satisfies our proposed requirements for a replacement of AUCPR. AUCNPR has
    the same same range for every data set, regardless of skew

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels or binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).

    recall_bounds : list of int or float, len=2, default=[0, 1]
        Lower and Upper bound of Recalls to calculate AUCPRmin over.
        Default defines whole recall space

    normalize_strategy : str
        Choose from 'average_precision', 'auc', or 'random'
        Average precision uses sklearn.metrics.auc method to calculating
        precision-recall curve auc, and average precision minimum
        AUC uses sklearn.metrics.auc method to calculating precision-recall
        curve auc, and aucpr_min
        Random uses sklearn.metrics.auc method to calculating precision-recall
        curve auc, and skew (random guess), which is not consistent across
        data sets. It can result in a negative score if an algorithm performs
        worse than random guessing, which seems counter-intuitive for an area
        under a curve

    Returns
    -------
    aucnpr : float
        Normalized Area Under Precision-Recall Curve

    References
    ----------
    .. [1] `Unachievable Region in Precision-Recall Space
            by Boyd, Costa, Davis, and Page 2012
            <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3858955/>

    See also
    --------
    precision_recall_curve : Compute precision-recall pairs for different
    probability thresholds

    average_precision_score : Area under the precision-recall curve

    auc : Area under curve

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import average_precision_score
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> normalized_aucpr(y_true, y_scores)
    0.7142857142857142...
    >>> normalized_aucpr(y_true, y_scores, normalize_strategy='auc')
    0.6994385331481325...
    >>> normalized_aucpr(y_true, y_scores, normalize_strategy='random')
    0.5833333333333333...
    >>> normalized_aucpr(y_true, y_scores, recall_bounds=[0.4,1],
                         normalize_strategy='auc')
    0.13553408061571867...
    """
    # Get AUC_PR and AUCPR_Min of classifier
    if normalize_strategy == 'average_precision':
        # Use average precision score values
        aucpr = average_precision_score(y_true, y_score)
        # Recall bounds not supported yet
        min_aucpr = min_average_precision(y_true)
        if recall_bounds != [0, 1]:
            print('\nRecall bounds not supported yet for Average Precision')
            print('Set to [0, 1]')
            recall_bounds = [0, 1]
    else:
        # Get arrays of preicisions and recalls
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # Get indices of when recall is above and between recall_bounds
        indices = [
            [i for i, e in enumerate(recall.tolist()) if e == x]
            for x in recall.tolist() if
            recall_bounds[0] <= x <= recall_bounds[1]
        ]
        # Turn list of lists into a list, and remove duplicates
        indices = list(set([item for sublist in indices for item in sublist]))
        # Get AUCPR of recall bounds
        aucpr = auc(recall[indices], precision[indices])
        if normalize_strategy == 'auc':
            # AUCPR minimum
            min_aucpr = aucpr_min(y_true, recall_bounds)
        if normalize_strategy == 'random':
            # Use skew
            min_aucpr = np.count_nonzero(y_true)/y_true.size
    # Calculate Noramlized Area Under Curve for Precision Recall Curve
    aucnpr = (aucpr - min_aucpr) \
        / (recall_bounds[1] - recall_bounds[0] - min_aucpr)
    return(aucnpr)
