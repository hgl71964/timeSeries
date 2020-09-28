# TimeSeries Project
-----

### Project Description
In this project, we aims to predict the price of cryptocurrency through time series forecasting. We would like to model the return of cryptocurrency price, which might have less fluctuation than directly predict price itself.

### Data Source
- Binance
- bitmex

### Data Sementation

We prepare time series data as walk-forward split:

![](https://raw.githubusercontent.com/Arturus/kaggle-web-traffic/master/images/split.png)


### Model Architecture 

##### In the *first* layer, we stack the following models to produce meta features:

Description: these models does not have sequential properties (or weak sequential); they can only encode one t and predict the next t, so they are stacked in the first layer to produce 'meta features'

- DNN: Deep Neural Network is a kind of function approximation.

- ARMA(2,2): ARMA is a classical time series model, which is essentially a Markov Chain with lieanr transformation matrix.

- XGBoost: XGBoost can score the importance of features, and it can also be applied to regression.

- Catboost: same as XGBoost, being a variant of gradient boost models. 


##### In the *second* layer, we implement a DNN based on the meta features from the first layer to predict the reture of the price.

Description: due to its inherent sequential property, we can easily encode a sequence with arbitrary length and produce another sequence with arbitrary length. 

- Seq2seq: seq2seq model modified for time series forecasting with attention; attention mechanism may be helpful for periodicity. Example of seq2seq model:

![](https://miro.medium.com/max/1400/1*CkeGXClZ5Xs0MhBc7xFqSA.png)


