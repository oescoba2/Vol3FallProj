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
    ptuna_available = False


class make_model():
    """This class
    
    Attributes
        - model_choice: the choice of model to make. The two options are
                            - 'clf' for classifier 
                            - 'reg' for regression.
                        This is defaulted to 'reg'.
        - model_type: the type of model to use for a given model choice.
                      The options are as follows:
                            - Classifiers (clf)
                                * 'rdf' for RandomForestClassifier (uses
                                oob_score)
                                * 'xgb' for  XGBClassifier
                            - Regression (reg):
                                * 'lin' for LinearRegression
                                * 'rdf' for RandomForestRegressor
                                * 'xgb' for XGBRegressor
        - params: the parameters to give the model. Defaulted to None
        - cv_fold: the number of folds to use in cross validation. Defa-
                   ulted to 2.
        - num_trials: the number of random samples or trials to use in 
                      RandomizedSearchCV or optuna, respectively. Def-
                      aulted to 300.
        - n_jobs: the number of parallel jobs to run when hypertuning.
                  Defaulted to -1 for all available cores.
        - tuning_srategy: the algorithm to use for hyperparameter tuning.
                          The options are as follows:
                            * 'grid' for GridSearchCV
                            * 'random' for RandomizedSearchCV
                            * 'bayesian' for Bayesian optimization using
                                TPESampler (Default of optuna)
    Methods
        - __init__(): the constructor for the class
        - make_quick_model():
        - make_usr_model():
        - make_full_model():

    Hidden Attributes:
        

    """

    def __init__(self, model_choice:str = "clf", model_type:str = "rdf", params:Dict = None, cv_fold:int = 2, tuning_strategy:str = "grid", num_trials:int=300, n_jobs:int=-1):
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
        if (model_type.strip().lower() != 'rdf') or (model_type.strip().lower() != 'xgb') or (model_type.strip().lower() != 'lin'):
            raise ValueError("Model type is not found. Please refer to the documentation to choose and appropriate model.")
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

        # Hidden attributes for random forest (Hyperparameter tuning)
        self._rdf_nestimators_range = (100, 301)  # Uses np.arange (so go one above)
        self._rdf_nestimators_step = 1
        self._rdf_maxdep_range = (3, 12)
        self._rdf_maxdep_step = 2
        self._rdf_minleaf_range = (2, 7)
        self._rdf_minleaf_step = 1
        self._rdf_minsamples_range = (2, 7)
        self._rdf_minsamples_step = 1

        # Hidden attributes for xgboost
        self._xgb_objective_clf = "multi:softmax"
        self._xgb_objective_num_classes = 10
        self._xgb_nestimators_range = (100, 301)
        self._xgb_nestimators_step = 1



        self._xgb_objective_reg = "reg:squarederror"


    def make_quick_model(self):
        """This function makes a ML model according to the given parameters that is
        meant to be used quickly and WILL NOT be hypertuned. It stores the model as
        an attribute. Ensure to only use this function to run quick tests on data 
        or make quick predictions. If no parameters are given, an out-of-the box 
        model is made.
        """

        warnings.warn(message="ACHTUNG! Your specified model will be made but cannot be hypertuned.")
        time.sleep(1.5)

        # Make classifiers
        if self.model_choice == "clf":
             
            # RandomForest
            if self.model_type == "rdf":
                if self.params is None:
                     self.model = RandomForestClassifier()         
                else:
                     self.model = RandomForestClassifier(**self.params) # The ** unpacks the dict as keyword args like key=value

            # XGBoost
            else:
                if self.params is None:
                    self.model = XGBClassifier()
                else:
                    self.model = XGBClassifier(**self.params)
        else:

            # RandomForest Regressor
            if self.model_type == "rdf":

                if self.params is None:
                    self.model = RandomForestRegressor()
                else:
                    self.model = RandomForestRegressor(**self.params) 
            
            # XGBoost Regressor
            elif self.model_type == 'xgb':
                if self.params is None:
                    self.model = XGBRegressor()
                else:
                    self.model = XGBRegressor(**self.params)

            # Linear Regression
            else:
                if self.params is None:
                    self.model = LinearRegression()
                else:
                    self.model = LinearRegression(**self.params)

    def make_usr_def_model(self, X_train:ArrayLike, y_train:ArrayLike):
        """This method makes a model according to user specified parameters,
        trains, and hypertunes according to those specifications. It does not
        consider other hyperparameters besides those given as a dictionary. En-
        sure that the params dictionary contains either a grid, distributions,
        or trial objects so that hypertuning can take place. 
        """

        if self.model_params is None:
            raise ValueError("Expected a dictionary for model_params but None was given. Please specify parameters using a dictionary")

        if self.tuning_strategy == "grid":

            # Make the classifier
            if self.model_choice == "clf":

                # RandomForest
                if self.model_type == "rdf":
                    print("Making user defined model RandomForest classifier with given parameters dictionary...")
                    clf = RandomForestClassifier(oob_score=True)
                    rdf_grid = GridSearchCV(estimator=clf, param_grid=self.model_params, n_jobs=self.n_jobs, cv=self.cv_fold, scoring = lambda est, X, y: est.oob_score_)
                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_grid.best_estimator_
                    self.model_params = rdf_grid.best_params_

                # XGBoost
                else:
                    print("Making user defined model XGBoost classifier with given parameters dictionary...")
                    objective = self.model_params["objective"]
                    clf = XGBClassifier(objective=objective)
                    xgb_grid = GridSearchCV(estimator=clf, param_grid=self.model_params, n_jobs=self.n_jobs, cv=self.cv_fold)
                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_grid.best_estimator_
                    self.model_params = xgb_grid.best_params_

            # Make the regressor
            else:

                # Make the regressor using RandomForest
                if self.model_type == "rdf":
                    print("Making user defined RandomForestRegressor with given parameters dictionary...")
                    reg = RandomForestRegressor(oob_score=True)
                    rdf_grid = GridSearchCV(estimator=reg, param_grid=self.model_params, n_jobs=self.n_jobs, cv=self.cv_fold,
                                            scoring = lambda est, X, y: est.oob_score_)
                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_grid.best_estimator_
                    self.model_params = rdf_grid.best_params_

                # Make the regressor using XGBoost
                elif self.model_type == "xgb":
                    print("Making user defined XGBoostRegressor with given parameters dictionary...")
                    objective = self.model_params["objective"]
                    clf = XGBRegressor(objective=objective)
                    xgb_grid = GridSearchCV(estimator=clf, param_grid=self.model_params, n_jobs=self.n_jobs, cv=self.cv_fold)
                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_grid.best_estimator_
                    self.model_params = xgb_grid.best_params_

                # Make the linear regression
                else:
                    print("Making making user defined Linear")
                    lin = LinearRegression(n_jobs=-1)
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
                    rdf_rand = RandomizedSearchCV(estimator=clf, param_distributions=self.model_params, n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials, scoring = lambda est, X, y: est.oob_score_)
                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_rand.best_estimator_
                    self.model_params = rdf_rand.best_params_

                else:
                    print("Making user defined model XGBoost classifier with given parameters dictionary...")
                    objective = self.model_params["objective"]
                    clf = XGBClassifier(objective=objective)
                    xgb_rand = RandomizedSearchCV(estimator=clf, param_distributions=self.model_params, n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials)
                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_rand.best_estimator_
                    self.model_params = xgb_rand.best_params_

            else:
                if self.model_type == "rdf":
                    print("Making user defined RandomForestRegressor with given parameters dictionary...")
                    reg = RandomForestRegressor(oob_score=True)
                    rdf_rand = RandomizedSearchCV(estimator=reg, param_distributions=self.model_params, n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials, scoring = lambda est, X, y: est.oob_score_)
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
                    objective = self.model_params["objective"]
                    clf = XGBRegressor(objective=objective)
                    xgb_grid = RandomizedSearchCV(estimator=clf, param_distributions=self.model_params, n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials)
                    print("Training and hypertuning using Exhaustive Search...")
                    xgb_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {xgb_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = xgb_grid.best_estimator_
                    self.model_params = xgb_grid.best_params_

                else:
                    print("Making making user defined Linear")
                    lin = LinearRegression(n_jobs=-1)
                    print("Training...")
                    lin.fit(X_train, y_train)
                    print("Training completed.")
                    print("Trained model saved as an attribute to class")
                    self.model = lin

        else:

            if not optuna_available:
                raise Exception("optuna module is not available. Please install in order to use bayesian optimization")

            if self.model_choice == "clf":

                def objective(trial:optuna.Trial) -> float:
                    """TODO"""
                    rdf_clf = RandomForestClassifier(**self.model_params, oob_score=True)
                    scores = cross_val_score(rdf_clf, X_train, y_train, scoring=lambda est, X, y: est.oob_score_, n_jobs=-1, cv=self.cv_fold)

                    return scores.mean()

                def get_optuna_model(trial):
                    """TODO"""

                    clf = RandomForestClassifier(**self.model_params, oob_score=True)
                    clf.fit(X_train, y_train)

                    return clf

                print("Training and tuning using Bayesian optimization...")
                study = optuna.create_study(direction="maximize", study_name="RandFor_tuning")
                study.optimize(objective, n_trials=self.num_trials, n_jobs=-1)
                print("Training completed.")

                print(f"The best oob_score is {study.best_value}")
                self.model_params = study.best_params
                self.model = get_optuna_model(study.best_trial)


    def make_full_model(self, X_train:ArrayLike, y_train:ArrayLike):
        """This methods makes a model in full
        
        """

        if self.model_params is not None:
            raise ValueError("Cannot make a model that trains and hypertunes on all available hyperparameters " +\
                             "when given user defined parameters. Use the hidden attributes to control model " +\
                             "specifications and don't pass in parameters to access full hyperparameters.")

        # Make grid search
        if self.tuning_strategy == "grid":
            
            # Make and hypertune classification models
            if self.model_choice == "clf":
                    
                if self.model_type == "rdf":

                    print("Now making RandomForestClassifier...")
                    clf = RandomForestClassifier(oob_score=True)
                    parameters = {"n_estimators": [int(x) for x in np.arange(*self._rdf_nestimators_range, step=self._rdf_nestimators_step)],
                                  "criterion": ["gini", "cross_entropy", "log_loss"],
                                  "max_depth": [int(x) for x in np.arange(*self._rdf_maxdep_range, step=self._rdf_maxdep_step)],
                                  "min_samples_leaf": [int(x) for x in np.arange(*self._rdf_minleaf_range, step=self._rdf_minleaf_step)],
                                  "min_samples_split": [int(x) for x in np.arange(*self._rdf_minsamples_range, step=self._rdf_minsamples_step)]
                                 }
                    rdf_grid = GridSearchCV(estimator=clf, param_grid=parameters, n_jobs=self.n_jobs, cv=self.cv_fold, scoring= lambda est, X, y: est.oob_score_)

                    print("Training and hypertuning using Exhaustive Search...")
                    rdf_grid.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_grid.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_grid.best_estimator_
                    self.model_params = rdf_grid.best_params_

                else:
                    print("Now making XGBoostClassifier...")
                    parameters = {"objective": "multi:softmax",
                                  "num_classes": self._xgb_objective_num_classes,
                                  "n_estimators": [int(x) for x in np.arange(*self._rdf_nestimators_range, step=self._rdf_nestimators_step)]
                    }


            # Make and hypertune regression models
            else:
                if self.model_type == "rdf":
                    pass

                elif self.model_type == "xgb":
                    pass

                else:
                    pass
                    

        # Train and tune using randomized search
        elif self.tuning_strategy == "random":

            # Make and hypertune classification models
            if self.model_choice == "clf":
                
                if self.model_type == "rdf":
                    print("Now making RandomForestClassifier...")
                    clf = RandomForestClassifier(oob_score=True)
                    distribs = {"n_estimators": [int(x) for x in np.arange(*self._rdf_nestimators_range, step=self._rdf_nestimators_step)],
                                "criterion": ["gini", "cross_entropy", "log_loss"],
                                "max_depth": [int(x) for x in np.arange(*self._rdf_maxdep_range, step=self._rdf_maxdep_step)],
                                "min_samples_leaf": [int(x) for x in np.arange(*self._rdf_minleaf_range, step=self._rdf_minleaf_step)],
                                "min_samples_split": [int(x) for x in np.arange(*self._rdf_minsamples_range, step=self._rdf_minsamples_step)],
                                }
                    rdf_rand = RandomizedSearchCV(estimator=clf, param_distributions=distribs, n_jobs=self.n_jobs, cv=self.cv_fold,
                                                  n_iter=self.num_trials, scoring=lambda est, X, y: est.oob_score_)

                    print("Training and hypertuning using Randomized Search...")
                    rdf_rand.fit(X_train, y_train)
                    print("Training completed.")

                    # Display and save best results
                    print(f"The best oob_score is {rdf_rand.best_score_}")
                    print("Best model and hyperparameters added as attribute to class.")
                    self.model = rdf_rand.best_estimator_
                    self.model_params = rdf_rand.best_params_

                else:
                    print("Now making XGBoostClassifier...")

                    clf = XGBClassifier(objective="multi:softmax")
                    distribs = {}

        # Train and tune using Bayesian optimization
        else:
            if not optuna_available:
                raise Exception("optuna module is not available. Please install in order to perform bayesian optimization.")


            def objective(trial:optuna.Trial) -> float:
                """This function accepts a trial object and creates and trains a Random
                ForestClassifier with the specified hpyerparameters as given by optuna 
                using TPESample Bayesian optimization algorithm.
                
                Parameters:
                    - trial (optuna.Trial): a specific trial object meant to signify the 
                                            current trial/model optuna is training
                
                Returns:
                    - (float): the mean of a cross-validated RandomForestClassifier
                """

                params = {"n_estimators": trial.suggest_int("n_estimators", *self._rdf_nestimators_range, self._rdf_nestimators_step),
                           "criterion": trial.suggest_categorical("criterion", ["gini", "cross_entropy", "log_loss"]),
                           "max_depth": trial.suggest_int("max_depth", *self._rdf_maxdep_range, self._rdf_maxdep_step),
                           "min_samples_leaf": trial.suggest_int("min_samples_leaf", *self._rdf_minleaf_range, self._rdf_minleaf_step),
                           "min_samples_split": trial.suggest_int(*self._rdf_minsamples_range, self._rdf_minsamples_step),
                          }
                rdf_clf = RandomForestClassifier(**params, oob_score=True)
                scores = cross_val_score(rdf_clf, X_train, y_train, scoring=lambda est, X, y: est.oob_score_, n_jobs=-1)

                return scores.mean()

            def get_optuna_model(trial:optuna.Trial) -> RandomForestClassifier:
                """This is a helper function meant to accept the best optuna trial
                in order to create"""

                params = {"n_estimators": trial.suggest_int("n_estimators", *self._rdf_nestimators_range, self._rdf_nestimators_step),
                        "criterion": trial.suggest_categorical("criterion", ["gini", "cross_entropy", "log_loss"]),
                        "max_depth": trial.suggest_int("max_depth", *self._rdf_maxdep_range, self._rdf_maxdep_step),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", *self._rdf_minleaf_range, self._rdf_minleaf_step),
                        "min_samples_split": trial.suggest_int(*self._rdf_minsamples_range, self._rdf_minsamples_step),
                        }

                clf = RandomForestClassifier(**params, oob_score=True)
                clf.fit(X_train, y_train)

                return clf

            study = optuna.create_study(direction="maximize", study_name="rand_for_tuning")
            print("Training and tuning using Bayesian optimization...")
            study.optimize(objective, n_trials=self.num_trials, n_jobs=-1)
            print("Training completed.")

            print(f"The best oob_score is {study.best_value}")
            self.model_params = study.best_params
            self.model = get_optuna_model(study.best_trial)





        