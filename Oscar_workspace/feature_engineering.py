from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class engineering():
    """This class is meant to perform feature engineering by either 
    selecting the best features according to mutual information or 
    generate new features using PCA (or a combination of both).
    
    Attributes:
        - X_tr: an array holding the training array of features
        - y_tr: an array holding the training targets
        - select_kfeats: a number specifying the number of features to
                         select
        - mut_info_kneighbors: the number of neighbors to use to compute
                               mutual information
        - pca_comp: the number of principal components to compute
        - pca_desired_var: the desired variance the principal components
                           should have
        - X_tr_scaled_selected: an array holding the selected and scaled
                                training array
        - features_selected: a list containing the names of the features 
                             selected
        - X_tr_pca: an array holding the transformed X_tr array into the 
                    principal components containing only a certain number
                    of components needed to reach a desired variance.

    Methods:
        - __init__(): the constructor for the class
        - get_bestk_features(): method to select a certain number of features
        - get_pca_features(): method to select a certain number of 
                              principal components
        - get_X_test_pca(): method to transform a test set of features into 
                            the space of principal components
                        
    
    Hidden Attributes:
        - _scaler: an instance of StandardScaler used to scale data. This 
                   is fitted with data once select_kfeats or get_pca_feat-
                   res are called.
        - _pca: an instance of PCA used to compute principal components. It
                is fitted once get_pca_features is called.
        - _X_tr_scaled: an array holding the scaled X_tr array using _scaler
        - _selector: an instance of SelectKBest that is fitted at the time
                     select_features is called
        _ _pca_mask: a Boolean array holding the True values of the principal
                     components that are needed to achieve a desired variance

    Hidden Methods:
        - _mutual_scorer(): method to compute the mutual information
    """

    def __init__(self, X_tr:NDArray, y_tr:ArrayLike, select_kfeats:int=20, 
                 mut_info_kneighbors:int=20, pca_comp:int=10,
                 pca_desired_var:float=0.95) -> None:
        
        """The constructor for the class. This functions accepts arguments
        and creates the attributes for the class.
        
        Parameters:
            - X_tr (NDArray): an array containing the features that will
                              be used for training.
            - y_tr (ArrayLike): an array containg the targets that will be
                                used for training.
            - select_kfeats (int): the number of features to select using
                                   mutual_info_regression
            - mut_info_kneighbors (int): the number of neighbors (K nearest)
                                         to use for approximating mutual 
                                         information
            - pca_comp (int): the number of principal components to compute
                              (using SVD)
            - pca_desired_var (float): the desired variance that the chosen
                                       number principal components must have
        
        Returns:
            None
        """

        self.X_tr = X_tr
        self.y_tr = y_tr
        self.select_kfeatures = select_kfeats
        self.mut_info_kneighbors = mut_info_kneighbors
        self.pca_components = pca_comp
        self.pca_desired_var = pca_desired_var
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=pca_comp)

        # Check input
        if (pca_desired_var > 1) or (pca_desired_var < 0.1):
            raise ValueError("Desired PCA variance cannot be greater than 1 or less than 0.1!")
    
    def _mutual_scorer(self, X:NDArray, y:ArrayLike) -> ArrayLike:
        """This function is meant to be called when selecting features
        according to mutual_info_regression. It allows the user to 
        give the argument of number of neighbors to use.

        Parameters:
            - X (NDArray): an array X that will have its mutual informa-
                           tion to y computed
            - y (ArrayLike): an array y used as the target with which
                             to measure the mutual information of X
        
        Returns:
            - (ArrayLike): the computed mutual information estimation
        """

        return mutual_info_regression(X, y, n_neighbors=self.mut_info_kneighbors)

    def get_bestk_features(self) -> None:
        """This function computes the mutual information between the stored
        training feature array X_tr and training target array y. It then
        selects a predetermined number of components and stores them as an
        attribute including the names.

        Parameters:
            - None

        Returns:
            - None
        """

        self._X_tr_scaled = self._scaler.fit_transform(X=self.X_tr)
        self._selector = SelectKBest(score_func=self._mutual_scorer, k=self.select_kfeatures).fit(X=self._X_tr_scaled, y=self.y_tr)
        self.features_selected = self.X_tr.columns[self._selector.get_support()]
        self.X_tr_scaled_selected = self._X_tr_scaled[:, self._selector.get_support()]

    def get_pca_features(self) -> None:
        """This function performs PCA and saves the number of principal
        components needed to reach a predetermined desired variance. If
        it cannot reach the desired variance, the function raises an
        Exception.

        Parameters:
            - None

        Returns:
            - None
        """

        self._X_tr_scaled = self._scaler.fit_transform(X=self.X_tr)
        X_pca = self._pca.fit_transform(X=self._X_tr_scaled)
        cum_sum = np.cumsum(self._pca.explained_variance_ratio_)          # Get the cumulative sum of the variance of each component

        # Check that the desired variance is actually met by the chosen number of components
        if cum_sum.max() < self.pca_desired_var:
            raise Exception(f"Desired variance is {self.pca_desired_var} but using {self.pca_components} components results in {cum_sum.max():.5f}")
        
        if cum_sum.min() > self.pca_desired_var:
            raise Exception(f"Minimum variance of PCA is {cum_sum.max():.5f} which is greater than {self.pca_desired_var}. Please specify a greater value.")
        
        self._pca_mask = (cum_sum >= self.pca_desired_var)                # Save the mask
        self.X_tr_pca = X_pca[:, ~self._pca_mask]          # Save the principal components needed to achieve desired variance as an attribute

    def get_X_test_pca(self, X_ts:NDArray) -> NDArray:
        """This function accepts an NDArray and returns the transformation of
        the array into the already created principle component space.

        Parameters:
            - X_ts (NDArray): an array containing the features that will be 
                              used for testing.
        
        Returns:
            - (NDArray): the transformed features array in the space of the 
                         computed principal components
        """

        return (self._pca.transform(X=X_ts))[:, ~self._pca_mask]
    
    
    def select_extract(self) -> None:

        if self.pca_components > self.select_kfeatures:
            raise ValueError(f"Cannot compute more principal components than you have features! "+\
                             f"Selected {self.select_select_kfeatures} features and computed {self.pca_components} principal components")
        
        # Select the K Best features according to mutual information
        self._X_tr_scaled = self._scaler.fit_transform(X=self.X_tr)
        self._selector = SelectKBest(score_func=self._mutual_scorer, k=self.select_kfeatures).fit(X=self._X_tr_scaled, y=self.y_tr)
        self.X_tr_scaled_selected = self._X_tr_scaled[:, self._selector.get_support()]
        self.features_selected = self.X_tr.columns[self._selector.get_support()]

        # Now perform PCA on the selected features
        X_pca = self._pca.fit_transform(X=self._X_tr_scaled)
        cum_sum = np.cumsum(self._pca.explained_variance_ratio_)

        if cum_sum.max() < self.pca_desired_var:
            raise Exception(f"Desired variance is {self.pca_desired_var} but using {self.pca_components} components results in {cum_sum.max():.5f}")
        if cum_sum.min() > self.pca_desired_var:
            raise Exception(f"Minimum variance of PCA is {cum_sum.max():.5f} which is greater than {self.pca_desired_var}. Please specify a greater value.")
        
        self._pca_mask = (cum_sum >= self.pca_desired_var)                
        self.X_tr_pca = X_pca[:, ~self._pca_mask]       
    
