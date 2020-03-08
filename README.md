# datagrasp
-----
### Collaborators:
- Guoliang HE (guoliang.he.19@.ucl.ac.uk)
- Chris Chia 
-----

### Project description
In this project, we aims to predict the price of cryptocurrency through time series forecasting.


### model architecture 
In the first layer, we stack the following models to produce meta features:

1. Seq2seq
2. XGBoost
3. ARMA

In the second layer, we implement a DNN based on the meta features from the first layer to predict the reture of the price.

