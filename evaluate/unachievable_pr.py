# -*- coding: utf-8 -*-
"""
scikit_xtend Machine Learning Library Extensions
Created on 2019

@author: Alex Xu <ayx2@case.edu>

Based on Unachievable Region in Precision-Recall Space Paper
"""

import numpy as np
from math import log
from sklearn.metrics import precision_recall_fscore_support

def min_precision(y_true, y_pred):
    """
    Precision (p) and recall (r) must satisfy,
    
    .. math::
        {p} \geq \\frac{{s}{r}}{1 - {s} + {s r}}
    where :math:`s` is the skew and :math:`r` is the recall
    
    Theorem 1 from Paper:
        Unachievable Region in Precision-Recall Space by Boyd, Costa, Davis, and Page 2012
        
    Parameters
    ----------
    y_true :
    y_pred :
        
    Returns
    -------
    References
    ----------
    .. [1] `Unachievable Region in Precision-Recall Space by Boyd, Costa, Davis, and Page 2012
    
    Examples
    --------
    >>> y_true = np.array([0, 1, 1, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 0, 0, 1])
    """
    # Get float or array of Recall Scores
    _, r, _, _ = precision_recall_fscore_support(y_true, y_pred)
    # Calculate skew
    s = np.count_nonzero(y_true)/y_true.size
    # Calculate minimum precision for given recall score and skew of labels
    min_precision = (s*r)/(1-s+s*r)
    return(min_precision)
    
def aucpr_min(y_true):
    """
    Minimum AUC for Precision Recall Curve, for given a skew between positive and negative labels.
    
    .. math::
        \\text{AUCPRmin} = \\ 1 + \\frac{(1 - {s}) \\ ln(1 - {s})}{s}
    where :math:`s` is the skew of the data
        
    Theorem 2 from Paper:
        Unachievable Region in Precision-Recall Space by Boyd, Costa, Davis, and Page 2012
        
    Parameters
    ----------
    y_true : array, shape = [n_samples,]
        True binary labels or binary label indicators. 1=Positive Labels, 0=Negative Labels
    
    Returns
    -------
    aucpr_min : float
    
    References
    ----------
    .. [1] `Unachievable Region in Precision-Recall Space by Boyd, Costa, Davis, and Page 2012
    
    Relevance
    ---------
    "The existence of the minimum AUCPR and minimum AP can affect the qualitative interpretation of a model's
    performance. For example, changing the skew of a data set from 0.01 to 0.5 (e.g., by subsampling the negative
    examples) increases the minimum AUCPR by approximately 0.3. (See Examples)
    
    "This leads to an automatic jump of 0.3 in AUCPR simply by changing the data set and with absolutely
    no change to the learning algorithm"
    
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
    # Calculate AUCPRmin
    aucpr_min = 1 + ( (1-s)*log(1-s) )/s
    return(aucpr_min)

def min_average_precision(y_true):
    """
    The minimum Average Precision (not to be confused with sklearn.metrics.average_precision_score), for positive
    (pos) and negative (neg) examples, respectively, is
    
    .. math::
        \\text{APmin} = \\frac{1}{pos} \\sum_{i=1}^{pos}\\frac{i}{i + {neg}}
        
    Theorem 3 from Paper:
        Unachievable Region in Precision-Recall Space by Boyd, Costa, Davis, and Page 2012
    
    Parameters
    ----------
    y_true : array, shape = [n_samples,]
        True binary labels or binary label indicators. 1=Positive Labels, 0=Negative Labels
    
    Returns
    -------
    min_average_precision : float
    
    References
    ----------
    .. [1] `Unachievable Region in Precision-Recall Space by Boyd, Costa, Davis, and Page 2012
    
    Relevance
    ---------
    "The existence of the minimum AUCPR and minimum AP can affect the qualitative interpretation of a model's
    performance. For example, changing the skew of a data set from 0.01 to 0.5 (e.g., by subsampling the negative
    examples) increases the minimum AUCPR by approximately 0.3. (See Examples)
    
    "This leads to an automatic jump of 0.3 in AUCPR simply by changing the data set and with absolutely
    no change to the learning algorithm"
    
    Examples
    --------
    >>> y_true = np.append(np.zeros(99),1) 
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
    i = np.arange(1,pos+1)
    min_average_precision = (1/pos) * np.sum( i/(i+neg) )
    
    return(min_average_precision)
    