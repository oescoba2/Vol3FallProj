from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class engineering():

    def __init__(self, X_tr:NDArray, y_tr:ArrayLike, mut_info_comp:int=20, 
                 mut_info_kneighbors:int=20, pca_comp:int=10,
                 pca_desired_var:float=0.95) -> None:

        self.X_tr = X_tr
        self.y_tr = y_tr
        self.mut_info_components = mut_info_comp
        self.mut_info_kneighbors = mut_info_kneighbors
        self.pca_components = pca_comp
        self.pca_desired_var = pca_desired_var
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=pca_comp)

        """
        if isinstance(X_tr, pd.DataFrame):
            self.pd_df = True

        else: 
            self.pd_df = False
        """

    def select_features(self) -> None:
        """
        
        """


        """
        mutual_info= mutual_info_regression(X=self.X_scaled, y=self.y_tr, n_neighbors=self.mut_info_kneighbors)
        mutual_info_ser = pd.Series(mutual_info)
        
        # Plot the pd.Dataframe
        if self.pd_df:
            mutual_info_ser.index = self.X_tr.columns
        else:
            labels = [f"feature {i+1}" for i in range(self.X_tr.shape[1])]
            mutual_info_ser.index = labels

        mutual_info_ser.sort_values(ascending=False).plot.bar()
        plt.show()
        """
        self._X_scaled = self._scaler.fit_transform(X=self.X_tr)

        def scorer(X, y, num_neighbors=self.mut_info_kneighbors):
            return mutual_info_regression(X,y, n_neighbors=num_neighbors)
        selector = SelectKBest(score_func=scorer, k=self.mut_info_components)
        selector.fit(X=self.X_scaled, y=self.y_tr)
        self.features_selected = self.X_tr.columns[selector.get_support()]
        self.X_tr_scaled = self._X_scaled[:, selector.get_support()]


    def extract_features(self) -> None:
        """
        """
        self._X_scaled = self._scaler.fit_transform(X=self.X_tr)
        self._pca.fit(X=self._X_scaled)
        



    




        

        



    