from old_geo_plotter import GeoPlotter
from numpy.typing import ArrayLike, NDArray
import joblib
import pandas as pd


class load_model():
    """
    
    """


    def __init__(self, path_to_model:str='./saved_models/rdf_reg_cv10_defArgs_MSE_343.sav', og_df:pd.DataFrame=[]):
        """
        
        Parameters:
            - path_to_model (str): the path to the joblib (.sav) file containing the trained model. Defaul-
                                   ted to 'rdf_reg_cv10_defArgs_MSE_343.sav' for a random forest regressor
                                   that used 10-fold cross-validation and had MSE of 0.343.
            - og_df (pd.DataFrame): the original dataframe unaltered from where all information was used to 
                                    make the ML model (containing features, world names, and targets). Def-
                                    aulted to an empty list.

        Returns:
            - None
        """

        self.model = joblib.load(path_to_model)
        if isinstance(og_df, list):
            raise ValueError("Must give the original dataframe when instatiating (i.e. constructor)!")
        self.df = og_df

    def get_happiness_predictions(self, X:NDArray|pd.DataFrame) -> None:
        """This function accepts a set data features and stores the predictions
        of happiness-scores made by the model as an attributes

        Parameters:
            - X (ArrayLike|pd.DataFrame): the set of data features
            
        Returns:
            - None
        """

        self.happiness_predictions = self.model.predict(X)

    def get_happiness_residuals(self, y_true:ArrayLike) -> None:
        """This function accepts the true target values and stores the
        residuals of the model.

        Parameters:
            - y_true (ArrayLike): the true target values

        Returns:
            - None
        """

        if self.happiness_predictions is None:
            raise Exception("Model has not yet computed happiness-score predictions. Compute the predictions using " +\
                            "'.get_happiness_predictions(X)' then rerun this method.")

        self.residuals = y_true - self.happiness_predictions

    
    def get_worldplot(self, y_data:ArrayLike=[], y_name:str="happiness predictions", fig_title:str='title', 
                      save_fig:bool=False, path_to_fig:str='fig.pdf') -> None:
        """This function plots the given data into the world plot. It then saves the 
        created image, if specified, into a pdf format.
        
        Parameters:
            - y_data (ArrayLike): the data to plot on the world map. Defaulted
                                  to an empty list. Default value will plot
                                  the happiness predictions.
            - y_name (str): the name of the column of the original dataframe
                            to plot on the worldmap. Defaulted to 'happiness
                            predictions'.
            - fig_title (str): the title to give the image. Defaulted to 'title'.
            - save_fig (bool): whether to save the figure into a pdf format 
                               or not. Defaulted to False
            - path_to_fig (str): the path, containing the filename, on where to save
                                 the figure. Defauled to 'fig.pdf'
        
        Returns:
            - None
        """

        if y_name.strip().lower() == "happiness predictions":
            y_data = self.happiness_predictions
        
        if (y_name.strip().lower() != "happiness predictions") and not y_data:
            raise ValueError("Argument 'y_data' cannot be empty if not using default 'happiness prediction'.")

        self.df[y_name] = y_data
        obj = GeoPlotter(df=self.df)
        obj.plot(col_name=y_name, title=fig_title, save_img=save_fig, img_name=path_to_fig)