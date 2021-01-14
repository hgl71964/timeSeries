import xgboost as xgb

"""
specify the learning task and the corresponding learning objective, and the objective options are below:
“reg:linear” –linear regression
“reg:logistic” –logistic regression
“binary:logistic” –logistic regression for binary classification, output probability
“binary:logitraw” –logistic regression for binary classification, output score before logistic transformation
“count:poisson” –poisson regression for count data, output mean of poisson distribution
max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
“multi:softmax” –set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
“multi:softprob” –same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
“rank:pairwise” –set XGBoost to do ranking task by minimizing the pairwise loss
"""

def train():
    return None
