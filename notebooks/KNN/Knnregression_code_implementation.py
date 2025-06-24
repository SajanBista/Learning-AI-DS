import numpy as np # Importing required module

class Knnregression():
    """

    K-NN based regression

    This is a K-NN regression built using Numpy module that
    only supports numerical data as input and euclidean distance
    for computing neighbors.
    """

    def __init__(self, k):
        '''
        Constructs K attribute of K-NN regression

        Parameters
        ----------
        k: int
          Number of neighbors to take into account in K-NN
        '''

        self.k = k

    def fit(self, X_train, y_train):
        '''
        Trains the K-NN model, i.e. storing the training dataset

        Parameters
        ----------
        X_train: Numpy.array, shape(n,m)
                Training feature matrix
        y_train: Numpy.array, shape(n,)
                Training target values
        '''

        self.X_train = X_train
        self.y_train = y_train

    def _calculate_euc_dist_mat(self, X_test):
        '''
        Computes the euclidean distance matrix between two feature vectors

        Parameters
        ----------
        X_test: Numpy.array, shape(z,m)
            New feature matrix, this feature vector and stored one's euclidean
            distance is computed.

        Returns
        -------
        euc_mat: Numpy.array, shape(n,z)
            Euclidean distance matrix, here element (1,1) is euclidean distance
            between stored data's 1st sample and new data's 1st sample, (1,2)
            between stored data's 1st sample and new data's 2nd sample and so on.
        '''
        a = np.sum(self.X_train**2, axis=1).reshape(-1,1) # Reshaping for proper euclidean matrix.
        b_T = np.sum(X_test**2, axis=1)
        W = -2 * np.dot(self.X_train,X_test.T)
        euc_mat = np.sqrt(a + b_T + W + 1e-10) # Adding small value to avoid warning.

        return euc_mat

    def predict(self, X_test):
        '''
        Predicts the target labels of provided data

        Parameters
        ----------
        X_test: Numpy.array, shape(z,m)

        Returns
        -------
        np.array(self.predictions): Numpy,array, shape(z,)
                Predictions of the target labels
        '''

        self.predictions = []

        dist_mat = self._calculate_euc_dist_mat(X_test)

        for i in range(X_test.shape[0]):
            distance = dist_mat[:,i] # Taking ith column of distance matrix
            near_neigh_index = np.argsort(distance)[:self.k]
            near_neigh_labels = self.y_train[near_neigh_index]

            self.predictions.append(np.mean(near_neigh_labels)) # Aggregation

        return np.array(self.predictions)