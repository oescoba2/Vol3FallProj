from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from typing import List, Tuple, Callable, ClassVar, Dict
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import time
import warnings

try:
    import optuna

    optuna_available = True
except ImportError:
    warnings.warn(message="Unable to import optuna. Bayesian optimization is not available.")
    optuna_available = False

class make_model():
    """This class is meant to create a user defined model. It allows the user to specify
    the model type, model choice, hyperparameters, and hyperparameter tuning strategy as
    well as other important options. The user can make various types of models that can
    be used trained and hypertuned or just instantiated. See the docstring of each method
    for more information.

    The class contains hidden attributes that are used to specify hyperparameters or
    hyperparameter ranges or step sizes. These are divided by model type and have
    distinction by model type. 'reg' is for regression and 'clf' is for classification.
    There are also hidden methods that are used for hyperparameter tuning using optuna.

    Refer to all of the docstrings for documentations on the class, attributes methods,
    and hidden attributes for more information.

    Attributes
        - model_choice: the choice of model to make. The two options are
                            - 'clf' for classifier
                            - 'reg' for regression.
                        This is defaulted to 'reg'.
        - model_type: the type of model to use for a given model choice.
                      This is defaulted to 'rdf' for RandomForest. The
                      options are as follows:
                            - Classifiers (clf)
                                * 'rdf' for RandomForestClassifier (uses
                                oob_score)
                                * 'xgb' for  XGBClassifier
                            - Regression (reg):
                                * 'lin' for LinearRegression
                                * 'rdf' for RandomForestRegressor
                                * 'xgb' for XGBRegressor
        - model_params: the parameters to give the model. Defaulted to None.
        - cv_fold: the number of folds to use in cross validation. Defa-
                   ulted to 2.
        - num_trials: the number of random samples or trials to use in
                      RandomizedSearchCV or optuna, respectively. Def-
                      aulted to 300.
        - n_jobs: the number of parallel jobs to run when hypertuning.
                  Defaulted to -1 for all available cores.
        - tuning_strategy: the algorithm to use for hyperparameter tuning.
                           Defaulted to 'grid'. The options are as follows:
                            * 'grid' for GridSearchCV
                            * 'random' for RandomizedSearchCV
                            * 'bayesian' for Bayesian optimization using
                               TPESampler (Default of optuna)
        - model: the model that is created and trained. If there is
                 hypertuning, the best model is stored here.
        - best_params: the best hyperparameters found during hypertuning.
                       Defaulted to None if no hypertuning is done.

    Methods
        - __init__(): the constructor for the class
        - make_quick_model(): makes a model that is meant to just be instantiated.
                              No hypertuning is done nor fitting is done.
        - make_usr_model(): makes, trains, and hypertunes  a model according to
                            user specifications
        - make_full_model(): makes, trains, and hypertunes  a model on various
                             available hyperparameters. It uses the hidden
                             attributes to control model specifications and
                             doesn't require the user to pass in parameters to
                             access full hyperparameters.

    Hidden Attributes:

        - RandomForest:
            * _rdf_nestimators_range: the range of n_estimators to use.
                                      Defaulted to (100, 301)
            * _rdf_nestimators_step: the step size for n_estimators.
                                      Defaulted to 1
            * _rdf_maxdep_range: the range of max_depth to use. Defaulted
                                 to (4, 15)
            * _rdf_maxdep_step: the step size for max_depth. Defaulted
                                to 2
            * _rdf_minleaf_range: the range of min_samples_leaf to use.
                                  Defaulted to (2, 7)
            * _rdf_minleaf_step: the step size for min_samples_leaf.
                                 Defaulted to 1
            * _rdf_minsamples_range: the range of min_samples_split to use.
                                     Defaulted to (2, 7)
            * _rdf_minsamples_step: the step size for min_samples_split.
                                    Defaulted to 1
            * _rdf_maxfeat_range: the range of max_features to use.
                                  Defaulted to (4, 15)
            * _rdf_maxfeat_step: the step size for max_features. Defaulted
                                 to 1

        - XGBoost:
            * _xgb_objective_clf: the objective for XGBoost classifier.
                                  Defaulted to "multi:softmax"
            * _xgb_num_classes_clf: the number of classes/labels for
                                    XGBoost classifier. Defaulted to 9
            * _xgb_nestimators_range: the range of n_estimators to use.
                                      Defaulted to (100, 301)
            * _xgb_nestimators_step: the step size for n_estimators.
                                     Defaulted to 1
            * _gxb_eta_range: the range of eta/learning rate to use.
                              Defaulted to (0.001, 0.1)
            * _alpha_range: the range of alpha to use (L1 regularization).
                            Defaulted to (0.6, 10.1)
            * _lambda_range: the range of lambda to use (L2 regularization).
                             Defaulted to (0.6, 10.1)
            * _gamma_range: the range of gamma to use (minimum loss reduction
                            or penalty for many leaves). Defaulted to (0.6, 10.1)
            * _xgb_max_depth_range: the range of max_depth to use. Defaulted
                                    to (3, 10)
            * _xgb_max_depth_step: the step size for max_depth. Defaulted to 1
            * _xgb_objective_reg: the objective for XGBoost regressor. Defaulted
                                  to "reg:squarederror"

    Hidden Methods:
        - _rdf_obj(): a hidden method used to train and hypertune a RandomForest
                      model using optuna.
        - _get_rdf(): a hidden method used to get the best RandomForest model
                      after hypertuning.
        - _xgb_obj(): a hidden method used to train and hypertune a XGBoost
                      model using optuna.
        - _get_xgb(): a hidden method used to get the best XGBoost model after
                      hypertuning.
    """

    def __init__(self, model_choice: str = "clf", model_type: str = "rdf", params: Dict = None, cv_fold: int = 2,
                 tuning_strategy: str = "grid", num_trials: int = 300, n_jobs: int = -1) -> None:
        """This function defines the a user defined supervised learning model
        according to the given choice

        Parameters:
            - model_choice (str): the choice of model to make.
            - model_type (str): the type of model to use for a given model choice.
            - params (Dict): the parameters to give the model. Defaulted to None.
            - cv_fold (int): the number of parallel jobs to run when hypertuning.
            - tuning_srategy: the algorithm to use for hyperparameter tuning.
            - num_trials (int): number of samples to use when performing baye-
                                 sian optimization or randomized search.
            - n_jobs: the number of parallel jobs to run when hypertuning
        """

        # Check user input
        if (model_choice.strip().lower() != "clf") or (model_choice.strip().lower() != "reg"):
            raise ValueError("Model type must be either 'clf' for classification or 'reg' for regression.")
        if (model_type.strip().lower() != 'rdf') or (model_type.strip().lower() != 'xgb') or (
                model_type.strip().lower() != 'lin'):
            raise ValueError(
                "Model type is not found. Please refer to the documentation to choose and appropriate model.")
        if (cv_fold is None) or (cv_fold < 2):
            raise TypeError("cv_fold must be of type int that is greater than or equal to 2.")
        if (tuning_strategy != "grid") or (tuning_strategy != "random") or (tuning_strategy != "bayesian"):
            raise TypeError("Hyperparameter tuning strategy must be either 'bayesian', 'grid', or 'random'.")
        if not isinstance(num_trials, int):
            raise TypeError("num_trials must be of type int")
        if not isinstance(n_jobs, int):
            raise TypeError("n_jobs must be of type int")

        # Define the attributes
        self.model_choice = model_choice.strip().lower()
        self.model_type = model_type.strip().lower()
        self.model_params = params
        self.cv_fold = cv_fold
        self.tuning_strategy = tuning_strategy.strip().lower()
        self.num_trials = num_trials
        self.n_jobs = n_jobs

        # Hidden attributes for random forest (Hyperparameter tunining)
        self._rdf_nestimators_range = (100, 301)  # Uses np.arange (so go one above)
        self._rdf_nestimators_step = 1
        self._rdf_maxdep_range = (4, 15)
        self._rdf_maxdep_step = 2
        self._rdf_minleaf_range = (2, 7)
        self._rdf_minleaf_step = 1
        self._rdf_minsamples_range = (2, 7)
        self._rdf_minsamples_step = 1
        self._rdf_maxfeat_range = (4, 15)
        self._rdf_maxfeat_step = 1

        # Hidden attributes for xgboost
        self._xgb_objective_clf = "multi:softmax"
        self._xgb_num_classes_clf = 9
        self._xgb_nestimators_range = (100, 301)
        self._xgb_nestimators_step = 1
        self._gxb_eta_range = (0.001, 0.1)
        self._alpha_range = (0.6, 10.1)
        self._lambda_range = (0.6, 10.1)
        self._gamma_range = (0.6, 10.1)
        self._xgb_max_depth_range = (3, 10)
        self._xgb_max_depth_step = 1
        self._xgb_objective_reg = "reg:squarederror"

    def make_quick_model(self) -> None:
        """This function makes a ML model according to the given parameters that is
        meant to be used quickly and WILL NOT be hypertuned. It stores the model as
        an attribute. Ensure to only use this function to run quick tests on data
        or make quick predictions. If no parameters are given, an out-of-the box
        model is made.
        """

        warnings.warn(message="ACHTUNG! Your specified model will be made but cannot be hypertuned.")
        time.sleep(1.5)

        self.best_params = None

        # Make classifiers
        if self.model_choice == "clf":

            # RandomForest
            if self.model_type == "rdf":
                if self.model_params is None:
                    self.model = RandomForestClassifier()
                else:
                    self.model = RandomForestClassifier(
                        **self.model_params)  # The ** unpacks the dict as keyword args like key=value

            # XGBoost
            else:
                if self.model_params is None:
                    self.model = XGBClassifier()
                else:
                    self.model = XGBClassifier(**self.model_params)
        else:

            # RandomForest Regressor
            if self.model_type == "rdf":

                if self.model_params is None:
                    self.model = RandomForestRegressor()
                else:
                    self.model = RandomForestRegressor(**self.model_params)

            # XGBoost Regressor
            elif self.model_type == 'xgb':
                if self.model_params is None:
                    self.model = XGBRegressor()
                else:
                    self.model = XGBRegressor(**self.model_params)

            # Linear Regression
            else:
                if self.model_params is None:
                    self.model = LinearRegression()
                else:
                    self.model = LinearRegression(n_jobs=-1)

    def make_usr_def_model(self, X_train: ArrayLike, y_train: ArrayLike) -> None:
        """This method makes a model according to user specified parameters,
        trains, and hypertunes according to those specifications. It stores
        the model as an attribute as well as the best parameters. It does
        not consider other hyperparameters besides those given as a dictionary.
        Ensure that the model_params dictionary contains either a grid,
        distributions, or trial objects so that hypertuning can take place.

        Parameters:
            - X_train (ArrayLike): the training data
            - y_train (ArrayLike): the target data
        """

        if self.model_params is None:
            raise ValueError(
                "Expected a dictionary for model_params but None was given. Please specify parameters using a dictionary")

        if self.tuning_strategy == "grid":

            # Make the classifier
            if self.model_choice == "clf":

                # RandomForest
                if self.model_type == "rdf":
                    print("Making user defined model RandomForest classifier with given parameters dictionary...")
                    clf = RandomForestClassifier(oob_score=True)
                    rdf_grid = GridSearchCV(estimator=clf, param_grid=self.model_params, n_jobs=self.n_jobs,
                                            cv=self.cv_fold, scoring=lambda est, X, y: est.oob_score_)
                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_grid.best_estimator_
                    self.best_params = rdf_grid.best_params_

                # XGBoost
                else:
                    print("Making user defined model XGBoost classifier with given parameters dictionary...")
                    xgb_grid = GridSearchCV(estimator=XGBClassifier(), param_grid=self.model_params, n_jobs=self.n_jobs,
                                            cv=self.cv_fold)
                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_grid.best_estimator_
                    self.best_params = xgb_grid.best_params_

            # Make the regressor
            else:

                # Make the regressor using RandomForest
                if self.model_type == "rdf":
                    print("Making user defined RandomForestRegressor with given parameters dictionary...")
                    reg = RandomForestRegressor(oob_score=True)
                    rdf_grid = GridSearchCV(estimator=reg, param_grid=self.model_params, n_jobs=self.n_jobs,
                                            cv=self.cv_fold,
                                            scoring=lambda est, X, y: est.oob_score_)
                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_grid.best_estimator_
                    self.best_params = rdf_grid.best_params_

                # Make the regressor using XGBoost
                elif self.model_type == "xgb":
                    print("Making user defined XGBoostRegressor with given parameters dictionary...")
                    xgb_grid = GridSearchCV(estimator=XGBRegressor(), param_grid=self.model_params, n_jobs=self.n_jobs,
                                            cv=self.cv_fold)
                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_grid.best_estimator_
                    self.best_params = xgb_grid.best_params_

                # Make the linear regression
                else:
                    print("Making making user defined Linear")
                    lin = LinearRegression(n_jobs=-1)
                    self.best_params = None
                    print("Training...")
                    lin.fit(X_train, y_train)
                    print("Training completed.")
                    print("Trained model saved as an attribute to class")
                    self.model = lin

        elif self.tuning_strategy == "random":

            if self.model_choice == "clf":
                if self.model_type == "rdf":
                    print("Making user defined model RandomForest classifier with given parameters dictionary...")
                    clf = RandomForestClassifier(oob_score=True)
                    rdf_rand = RandomizedSearchCV(estimator=clf, param_distributions=self.model_params,
                                                  n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials, scoring=lambda est, X, y: est.oob_score_)
                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_rand.best_estimator_
                    self.best_params = rdf_rand.best_params_

                # XGBoost
                else:
                    print("Making user defined model XGBoost classifier with given parameters dictionary...")
                    xgb_rand = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=self.model_params,
                                                  n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials)
                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best score is {xgb_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_rand.best_estimator_
                    self.best_params = xgb_rand.best_params_

            else:
                if self.model_type == "rdf":
                    print("Making user defined RandomForestRegressor with given parameters dictionary...")
                    reg = RandomForestRegressor(oob_score=True)
                    rdf_rand = RandomizedSearchCV(estimator=reg, param_distributions=self.model_params,
                                                  n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials, scoring=lambda est, X, y: est.oob_score_)
                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_rand.best_estimator_
                    self.model_params = rdf_rand.best_params_

                elif self.model_type == "xgb":
                    print("Making user defined XGBoostRegressor with given parameters dictionary...")
                    xgb_grid = RandomizedSearchCV(estimator=XGBRegressor(), param_distributions=self.model_params,
                                                  n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials)
                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best score is {xgb_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_grid.best_estimator_
                    self.model_params = xgb_grid.best_params_

                # Linear Regression
                else:
                    print("Making making user defined Linear")
                    lin = LinearRegression(n_jobs=-1)
                    self.best_params = None
                    print("Training...")
                    lin.fit(X_train, y_train)
                    print("Training completed.")
                    print("Trained model saved as an attribute to class")
                    self.model = lin

        else:

            if not optuna_available:
                raise Exception("optuna module is not available. Please install in order to use bayesian optimization")

            if self.model_choice == "clf":

                if self.model_type == "rdf":
                    def obj(trial: optuna.Trial) -> float:
                        """This is a wrapper function n meant to call the hidden _rdf_obj method.
                        Optuna requires a function call without any args"""
                        return self._rdf_obj(trial, X_train, y_train)

                    print("Training and tuning using Bayesian optimization...")
                    study = optuna.create_study(direction="maximize", study_name="RandFor_tuning")
                    study.optimize(obj, n_trials=self.num_trials, n_jobs=-1)
                    print("Training completed.")

                    print(f"The best oob_score is {study.best_value}")
                    self.best_params = study.best_params
                    self.model = self._get_rdf(study.best_trial, X_train, y_train)

                else:
                    def obj(trial: optuna.Trial) -> float:
                        """This is a wrapper function n meant to call the hidden _rdf_obj method.
                        Optuna requires a function call without any args"""
                        return self._xgb_obj(trial, X_train, y_train)

                    print("Training and tuning using Bayesian optimization...")
                    study = optuna.create_study(direction="maximize", study_name="XGB_tuning")
                    study.optimize(obj, n_trials=self.num_trials, n_jobs=-1)
                    print("Training completed.")

                    print(f"The best score is {study.best_value}")
                    self.best_params = study.best_params
                    self.model = self._get_xgb(study.best_trial, X_train, y_train)

            # Make regressors
            else:
                if self.model_type == "rdf":
                    def obj(trial: optuna.Trial) -> float:
                        """This is a wrapper function n meant to call the hidden _rdf_obj method.
                        Optuna requires a function call without any args"""
                        return self._rdf_obj(trial, X_train, y_train)

                    print("Training and tuning using Bayesian optimization...")
                    study = optuna.create_study(direction="maximize", study_name="RandFor_tuning")
                    study.optimize(obj, n_trials=self.num_trials, n_jobs=-1)
                    print("Training completed.")

                    print(f"The best oob_score is {study.best_value}")
                    self.best_params = study.best_params
                    self.model = self._get_rdf(study.best_trial, X_train, y_train)

                elif self.model_type == "xgb":
                    def obj(trial: optuna.Trial) -> float:
                        """This is a wrapper function n meant to call the hidden _rdf_obj method.
                        Optuna requires a function call without any args"""
                        return self._xgb_obj(trial, X_train, y_train)

                    print("Training and tuning using Bayesian optimization...")
                    study = optuna.create_study(direction="maximize", study_name="XGB_tuning")
                    study.optimize(obj, n_trials=self.num_trials, n_jobs=-1)
                    print("Training completed.")

                    print(f"The best score is {study.best_value}")
                    self.best_params = study.best_params
                    self.model = self._get_xgb(study.best_trial, X_train, y_train)

                else:
                    print("Making making user defined Linear")
                    lin = LinearRegression(n_jobs=-1)
                    print("Training...")
                    lin.fit(X_train, y_train)
                    print("Training completed.")
                    print("Trained model saved as an attribute to class")
                    self.model = lin
                    self.best_params = None

    # Hidden methods for optuna hyperparameter tuning
    def _rdf_obj(self, trial: optuna.Trial, X_train: ArrayLike, y_train: ArrayLike) -> float:
        """This function accepts a trial object and creates and trains a Random
        ForestClassifier with the specified hpyerparameters as given by optuna
        using TPESample Bayesian optimization algorithm.

        Parameters:
            - trial (optuna.Trial): a specific trial object meant to signify the
                                     current trial/model optuna is training
            - X_train (ArrayLike): the training data
            - y_train (ArrayLike): the target data

        Returns:
            - (float): the mean of a cross-validated RandomForestClassifier
        """

        if self.model_params is not None:
            params = {"n_estimators": trial.suggest_int("n_estimators", *self._rdf_nestimators_range,
                                                        self._rdf_nestimators_step),
                      "criterion": trial.suggest_categorical("criterion", ["gini", "cross_entropy", "log_loss"]),
                      "max_depth": trial.suggest_int("max_depth", *self._rdf_maxdep_range, self._rdf_maxdep_step),
                      "min_samples_leaf": trial.suggest_int("min_samples_leaf", *self._rdf_minleaf_range,
                                                            self._rdf_minleaf_step),
                      "min_samples_split": trial.suggest_int(*self._rdf_minsamples_range, self._rdf_minsamples_step),
                      "max_features": trial.suggest_int(*self._rdf_maxfeat_range, self._rdf_maxfeat_step)
                      }
        else:
            params = self.model_params

        # Make the model
        if self.model_choice == "clf":
            model = RandomForestClassifier(**params, oob_score=True)
        else:
            model = RandomForestRegressor(**params, oob_score=True)

        scores = cross_val_score(model, X_train, y_train, scoring=lambda est, X, y: est.oob_score_, n_jobs=self.n_jobs,
                                 cv=self.cv_fold)

        return scores.mean()

    def _get_rdf(self, trial: optuna.Trial, X_train: ArrayLike,
                 y_train: ArrayLike) -> RandomForestClassifier | RandomForestRegressor:
        """This is a helper function meant to accept the best optuna trial
        and return the best RandomForest model.

        Parameters:
            - trial (optuna.Trial): the best trial object from optuna
            - X_train (ArrayLike): the training data
            - y_train (ArrayLike): the target data

        Returns:
            - (RandomForestClassifier): the best RandomForestClassifier model
        """

        params = {
            "n_estimators": trial.suggest_int("n_estimators", *self._rdf_nestimators_range, self._rdf_nestimators_step),
            "criterion": trial.suggest_categorical("criterion", ["gini", "cross_entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", *self._rdf_maxdep_range, self._rdf_maxdep_step),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", *self._rdf_minleaf_range, self._rdf_minleaf_step),
            "min_samples_split": trial.suggest_int(*self._rdf_minsamples_range, self._rdf_minsamples_step),
            "max_features": trial.suggest_int(*self._rdf_maxfeat_range, self._rdf_maxfeat_step)
            }

        # Make the model
        if self.model_choice == "clf":
            model = RandomForestClassifier(**params, oob_score=True)
        else:
            model = RandomForestRegressor(**params, oob_score=True)

        model.fit(X_train, y_train)

        return model

    def _xgb_obj(self, trial: optuna.trial, X_train, y_train) -> float:
        """This function accepts an optuna trial module that indicates the
        current trial of hyperparameter tuning and creates a XGBoost model to
        train. It returns the score after having trained the classifier or
        regressor. This function serves as a single call during each
        trial by the study object.

        Parameters:
            - trial (optuna.trial): the current trial of hyperparameter tuning
            - X_train (ArrayLike): the training data
            - y_train (ArrayLike): the target data

        Returns:
            - (float): the classifier score on the testing dataset
        """

        if self.model_params is not None:
            params = {"n_estimators": trial.suggest_int("n_estimators", *self._xgb_nestimators_range,
                                                        self._xgb_nestimators_step),
                      "eta": trial.suggest_float("eta", *self._gxb_eta_range, log=True),
                      "alpha": trial.suggest_float("alpha", *self._alpha_range),
                      "lambda": trial.suggest_float("lambda", *self._lambda_range),
                      "gamma": trial.suggest_float("gamma", *self._gamma_range),
                      "max_depth": trial.suggest_int("max_depth", *self._xgb_max_depth_range, self._xgb_max_depth_step),
                      }
        else:
            params = self.model_params

        # Make the model
        if self.model_choice == "clf":
            params["objective"] = self._xgb_objective_clf
            params["num_classes"] = self._xgb_num_classes_clf
            model = XGBClassifier(**params)
        else:
            params["objective"] = self._xgb_objective_reg
            model = XGBRegressor(**params)

        scores = cross_val_score(model, X_train, y_train, n_jobs=self.n_jobs, cv=self.cv_fold)

        return scores.mean()

    def _get_xgb(self, trial: optuna.Trial, X_train: ArrayLike, y_train: ArrayLike) -> XGBClassifier | XGBRegressor:
        """This function accepts the best trial from optuna and returns the
        best XGBoost model according to the given hyperparameters.

        Parameters:
        - trial (optuna.Trial): the best trial from optuna
        - X_train (ArrayLike): the training data
        - y_train (ArrayLike): the target data

        Returns:
        - (XGBClassifier or XGBRegressor): the best XGBoost model
        """

        if self.model_params is not None:
            params = {"n_estimators": trial.suggest_int("n_estimators", *self._xgb_nestimators_range,
                                                        self._xgb_nestimators_step),
                      "eta": trial.suggest_float("eta", *self._gxb_eta_range, log=True),
                      "alpha": trial.suggest_float("alpha", *self._alpha_range),
                      "lambda": trial.suggest_float("lambda", *self._lambda_range),
                      "gamma": trial.suggest_float("gamma", *self._gamma_range),
                      "max_depth": trial.suggest_int("max_depth", *self._xgb_max_depth_range, self._xgb_max_depth_step),
                      }
        else:
            params = self.model_params

        # Make the model
        if self.model_choice == "clf":
            params["objective"] = self._xgb_objective_clf
            params["num_classes"] = self._xgb_num_classes_clf
            model = XGBClassifier(**params)
        else:
            params["objective"] = self._xgb_objective_reg
            model = XGBRegressor(**params)

        model.fit(X_train, y_train)

        return model

    def make_full_model(self, X_train: ArrayLike, y_train: ArrayLike) -> None:
        """This methods makes a model that trains and hypertunes on
        all available hyperparameters. It uses the hidden attributes
        to control model specifications and doesn't require the user
        to pass in parameters to access full hyperparameters. It
        stores the model as an attribute.

        Parameters:
            - X_train (ArrayLike): the training data
            - y_train (ArrayLike): the target data
        """

        if self.model_params is not None:
            raise ValueError("Cannot make a model that trains and hypertunes on all available hyperparameters " + \
                             "when given user defined parameters. Use the hidden attributes to control model " + \
                             "specifications and don't pass in parameters to access full hyperparameters.")

        # Make grid search
        if self.tuning_strategy == "grid":

            # Make and hypertune classification models
            if self.model_choice == "clf":

                if self.model_type == "rdf":

                    print("Now making RandomForestClassifier...")
                    clf = RandomForestClassifier(oob_score=True)
                    parameters = {"n_estimators": [int(x) for x in np.arange(*self._rdf_nestimators_range,
                                                                             step=self._rdf_nestimators_step)],
                                  "criterion": ["gini", "cross_entropy", "log_loss"],
                                  "max_depth": [int(x) for x in
                                                np.arange(*self._rdf_maxdep_range, step=self._rdf_maxdep_step)],
                                  "min_samples_leaf": [int(x) for x in np.arange(*self._rdf_minleaf_range,
                                                                                 step=self._rdf_minleaf_step)],
                                  "min_samples_split": [int(x) for x in np.arange(*self._rdf_minsamples_range,
                                                                                  step=self._rdf_minsamples_step)]
                                  }
                    rdf_grid = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=self.n_jobs, cv=self.cv_fold,
                                            scoring=lambda est, X, y: est.oob_score_)

                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_grid.best_estimator_
                    self.best_params = rdf_grid.best_params_

                else:
                    print("Now making XGBClassifier...")
                    params = {"n_estimators": [int(x) for x in
                                               np.arange(*self._xgb_nestimators_range, self._xgb_nestimators_step)],
                              "eta": [float(x) for x in np.linspace(*self._gxb_eta_range, num=100)],
                              "alpha": [float(x) for x in np.linspace(*self._alpha_range, num=100)],
                              "lambda": [float(x) for x in np.linspace(*self._lambda_range, num=100)],
                              "gamma": [float(x) for x in np.linspace(*self._gamma_range, num=100)],
                              "max_depth": [int(x) for x in
                                            np.arange(*self._xgb_max_depth_range, self._xgb_max_depth_step)],
                              "objective": self._xgb_objective_clf,
                              "num_classes": self._xgb_num_classes_clf
                              }
                    xgb_grid = GridSearchCV(estimator=XGBClassifier(), param_grid=params, n_jobs=self.n_jobs,
                                            cv=self.cv_fold)

                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_grid.best_estimator_
                    self.best_params = xgb_grid.best_params_

            # Make and hypertune regression models
            else:
                if self.model_type == "rdf":
                    print("Now making RandomForestRegressor...")
                    reg = RandomForestRegressor(oob_score=True)
                    parameters = {"n_estimators": [int(x) for x in np.arange(*self._rdf_nestimators_range,
                                                                             step=self._rdf_nestimators_step)],
                                  "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                                  "max_depth": [int(x) for x in
                                                np.arange(*self._rdf_maxdep_range, step=self._rdf_maxdep_step)],
                                  "min_samples_leaf": [int(x) for x in np.arange(*self._rdf_minleaf_range,
                                                                                 step=self._rdf_minleaf_step)],
                                  "min_samples_split": [int(x) for x in np.arange(*self._rdf_minsamples_range,
                                                                                  step=self._rdf_minsamples_step)],
                                  "max_features": [int(x) for x in
                                                   np.arange(*self._rdf_maxfeat_range, step=self._rdf_maxfeat_step)],
                                  }
                    rdf_grid = GridSearchCV(estimator=reg, param_grid=parameters, n_jobs=self.n_jobs, cv=self.cv_fold,
                                            scoring=lambda est, X, y: est.oob_score_)

                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_grid.best_estimator_
                    self.best_params = rdf_grid.best_params_

                elif self.model_type == "xgb":
                    print("Now making XGBRegressor...")
                    params = {"n_estimators": [int(x) for x in
                                               np.arange(*self._xgb_nestimators_range, self._xgb_nestimators_step)],
                              "eta": [float(x) for x in np.linspace(*self._gxb_eta_range, num=100)],
                              "alpha": [float(x) for x in np.linspace(*self._alpha_range, num=100)],
                              "lambda": [float(x) for x in np.linspace(*self._lambda_range, num=100)],
                              "gamma": [float(x) for x in np.linspace(*self._gamma_range, num=100)],
                              "max_depth": [int(x) for x in
                                            np.arange(*self._xgb_max_depth_range, self._xgb_max_depth_step)],
                              "objective": self._xgb_objective_reg,
                              }
                    xgb_grid = GridSearchCV(estimator=XGBRegressor(), param_grid=params, n_jobs=self.n_jobs,
                                            cv=self.cv_fold)

                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_grid.best_estimator_
                    self.best_params = xgb_grid.best_params_

                else:
                    print("Making making LinearRegression")
                    lin = LinearRegression(n_jobs=-1)
                    print("Training...")
                    lin.fit(X_train, y_train)
                    print("Training completed.")
                    print("Trained model saved as an attribute to class")
                    self.model = lin
                    self.best_params = None

        # Train and tune using randomized search
        elif self.tuning_strategy == "random":

            # Make and hypertune classification models
            if self.model_choice == "clf":

                if self.model_type == "rdf":
                    print("Now making RandomForestClassifier...")
                    clf = RandomForestClassifier(oob_score=True)
                    distribs = {"n_estimators": [int(x) for x in np.arange(*self._rdf_nestimators_range,
                                                                           step=self._rdf_nestimators_step)],
                                "criterion": ["gini", "cross_entropy", "log_loss"],
                                "max_depth": [int(x) for x in
                                              np.arange(*self._rdf_maxdep_range, step=self._rdf_maxdep_step)],
                                "min_samples_leaf": [int(x) for x in
                                                     np.arange(*self._rdf_minleaf_range, step=self._rdf_minleaf_step)],
                                "min_samples_split": [int(x) for x in np.arange(*self._rdf_minsamples_range,
                                                                                step=self._rdf_minsamples_step)],
                                "max_features": [int(x) for x in
                                                 np.arange(*self._rdf_maxfeat_range, step=self._rdf_maxfeat_step)],
                                }
                    rdf_rand = RandomizedSearchCV(estimator=clf, param_distributions=distribs, n_jobs=self.n_jobs,
                                                  cv=self.cv_fold,
                                                  n_iter=self.num_trials, scoring=lambda est, X, y: est.oob_score_)

                    print("Training and hypertuning using Randomized Search...")
                    rdf_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_rand.best_estimator_
                    self.best_params = rdf_rand.best_params_

                else:
                    print("Now making XGBClassifier...")
                    params = {"n_estimators": [int(x) for x in
                                               np.arange(*self._xgb_nestimators_range, self._xgb_nestimators_step)],
                              "eta": [float(x) for x in np.linspace(*self._gxb_eta_range, num=100)],
                              "alpha": [float(x) for x in np.linspace(*self._alpha_range, num=100)],
                              "lambda": [float(x) for x in np.linspace(*self._lambda_range, num=100)],
                              "gamma": [float(x) for x in np.linspace(*self._gamma_range, num=100)],
                              "max_depth": [int(x) for x in
                                            np.arange(*self._xgb_max_depth_range, self._xgb_max_depth_step)],
                              "objective": self._xgb_objective_clf,
                              "num_classes": self._xgb_num_classes_clf
                              }
                    xgb_rand = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=params,
                                                  n_jobs=self.n_jobs,
                                                  cv=self.cv_fold, n_iter=self.num_trials)

                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_rand.best_estimator_
                    self.best_params = xgb_rand.best_params_

            else:
                if self.model_type == "rdf":
                    print("Now making RandomForestRegressor...")
                    reg = RandomForestRegressor(oob_score=True)
                    parameters = {"n_estimators": [int(x) for x in np.arange(*self._rdf_nestimators_range,
                                                                             step=self._rdf_nestimators_step)],
                                  "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                                  "max_depth": [int(x) for x in
                                                np.arange(*self._rdf_maxdep_range, step=self._rdf_maxdep_step)],
                                  "min_samples_leaf": [int(x) for x in np.arange(*self._rdf_minleaf_range,
                                                                                 step=self._rdf_minleaf_step)],
                                  "min_samples_split": [int(x) for x in np.arange(*self._rdf_minsamples_range,
                                                                                  step=self._rdf_minsamples_step)],
                                  "max_features": [int(x) for x in
                                                   np.arange(*self._rdf_maxfeat_range, step=self._rdf_maxfeat_step)],
                                  }
                    rdf_rand = RandomizedSearchCV(estimator=reg, param_distributions=parameters, n_jobs=self.n_jobs,
                                                  cv=self.cv_fold,
                                                  n_iter=self.num_trials, scoring=lambda est, X, y: est.oob_score_)

                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_rand.best_estimator_
                    self.best_params = rdf_rand.best_params_

                elif self.model_type == "xgb":
                    print("Now making XGBRegressor...")
                    params = {"n_estimators": [int(x) for x in
                                               np.arange(*self._xgb_nestimators_range, self._xgb_nestimators_step)],
                              "eta": [float(x) for x in np.linspace(*self._gxb_eta_range, num=100)],
                              "alpha": [float(x) for x in np.linspace(*self._alpha_range, num=100)],
                              "lambda": [float(x) for x in np.linspace(*self._lambda_range, num=100)],
                              "gamma": [float(x) for x in np.linspace(*self._gamma_range, num=100)],
                              "max_depth": [int(x) for x in
                                            np.arange(*self._xgb_max_depth_range, self._xgb_max_depth_step)],
                              "objective": self._xgb_objective_reg,
                              }
                    xgb_rand = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=params,
                                                  n_jobs=self.n_jobs,
                                                  cv=self.cv_fold, n_iter=self.num_trials)

                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_rand.best_estimator_
                    self.best_params = xgb_rand.best_params_

                else:
                    print("Making making LinearRegression")
                    lin = LinearRegression(n_jobs=-1)
                    print("Training...")
                    lin.fit(X_train, y_train)
                    print("Training completed.")
                    print("Trained model saved as an attribute to class")
                    self.model = lin
                    self.best_params = None

        # Train and tune using Bayesian optimization
        else:
            if not optuna_available:
                raise Exception(
                    "optuna module is not available. Please install in order to perform bayesian optimization.")

            # Make classifiers
            if self.model_choice == "clf":
                if self.model_type == "rdf":

                    def obj(trial: optuna.Trial) -> float:
                        """This is a wrapper function n meant to call the hidden _rdf_obj method.
                        Optuna requires a function call without any args"""
                        return self._rdf_obj(trial, X_train, y_train)

                    print("Making RandomForestClassifier using Bayesian optimization...")
                    study = optuna.create_study(direction="maximize", study_name="randfor_clf_tuning")
                    print("Training and tuning using Bayesian optimization...")
                    study.optimize(obj, n_trials=self.num_trials, n_jobs=-1)
                    print("Training completed.")

                    print(f"The best oob_score is {study.best_value}")
                    self.best_params = study.best_params
                    self.model = self._get_rdf(study.best_trial, X_train, y_train)

                # XGBoost
                else:
                    def obj(trial: optuna.Trial) -> float:
                        """This is a wrapper function n meant to call the hidden _xgb_obj method.
                        Optuna requires a function call without any args"""
                        return self._xgb_obj(trial, X_train, y_train)

                    print("Making XGBClassifier using Bayesian optimization...")
                    study = optuna.create_study(direction="maximize", study_name="xgb_clf_tuning")
                    print("Training and tuning using Bayesian optimization...")
                    study.optimize(obj, n_trials=self.num_trials, n_jobs=-1)
                    print("Training completed.")

                    print(f"The best score is {study.best_value}")
                    self.best_params = study.best_params
                    self.model = self._get_xgb(study.best_trial, X_train, y_train)

            # Make regressors
            else:
                if self.model_type == "rdf":

                    def obj(trial: optuna.Trial) -> float:
                        """This is a wrapper function n meant to call the hidden _rdf_obj method.
                        Optuna requires a function call without any args"""
                        return self._rdf_obj(trial, X_train, y_train)

                    print("Making RandomForestClassifier using Bayesian optimization...")
                    study = optuna.create_study(direction="maximize", study_name="randfor_reg_tuning")
                    print("Training and tuning using Bayesian optimization...")
                    study.optimize(obj, n_trials=self.num_trials, n_jobs=-1)
                    print("Training completed.")

                    print(f"The best oob_score is {study.best_value}")
                    self.best_params = study.best_params
                    self.model = self._get_rdf(study.best_trial, X_train, y_train)

                elif self.model_type == "xgb":
                    def obj(trial: optuna.Trial) -> float:
                        """This is a wrapper function n meant to call the hidden _rdf_obj method.
                        Optuna requires a function call without any args"""
                        return self._xgb_obj(trial, X_train, y_train)

                    print("Making XGBRegressor using Bayesian optimization...")
                    study = optuna.create_study(direction="maximize", study_name="xgb_reg_tuning")
                    print("Training and tuning using Bayesian optimization...")
                    study.optimize(obj, n_trials=self.num_trials, n_jobs=-1)
                    print("Training completed.")

                    print(f"The best score is {study.best_value}")
                    self.best_params = study.best_params
                    self.model = self._get_xgb(study.best_trial, X_train, y_train)

                else:
                    print("Making making LinearRegression")
                    lin = LinearRegression(n_jobs=-1)
                    print("Training...")
                    lin.fit(X_train, y_train)
                    print("Training completed.")
                    print("Trained model saved as an attribute to class")
                    self.model = lin
                    self.best_params = None