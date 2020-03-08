# DataGrasp
-----
### Collaborators:
- Guoliang HE (guoliang.he.19@.ucl.ac.uk)
- Chris Chia 
-----

### Project Description
In this project, we aims to predict the price of cryptocurrency through time series forecasting. We would like to model the return of cryptocurrency price, which might have less fluctuation than directly predict price itself.

### Data Source
- Binance


### Model Architecture 
In the first layer, we stack the following models to produce meta features:

- Seq2seq
- XGBoost
- ARMA
- Catboost
- 1-d convolutional neural network

In the second layer, we implement a DNN based on the meta features from the first layer to predict the reture of the price.

