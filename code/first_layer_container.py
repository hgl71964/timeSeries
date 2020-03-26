import numpy as np


class first_layer_container:

    '''
    this is a container than ensemble the first layer results
    '''

    def __init__(self):
        self.counter = 0

    def append(self, meta_train, meta_test):

        if self.counter == 0:
            self.meta_train = meta_train.reshape(1, -1)
            self.meta_test = meta_test.reshape(1, -1)

        else:
            self.meta_train = np.concatenate(
                [self.meta_train, meta_train.reshape(1, -1)], axis=0)
            self.meta_test = np.concatenate(
                [self.meta_test, meta_test.reshape(1, -1)], axis=0)

        self.counter += 1

    def ensemble_results(self):

        return self.meta_train.mean(axis=0), self.meta_test.mean(axis=0)
