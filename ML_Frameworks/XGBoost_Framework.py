import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

class XGBoost_Regression_Framework:
    """
    A class to make XGBoost training and estimating easily for mostly Kaggle Competitions 

    Attributes
    ----------
    X : pd.Dataframe
        Train Dataset

    y : pd.Dataframe
        Train labels 
    
    test_set : pd.Dataframe
        Test Dataset but it can be null if there is no test set.
        This test set is not the Validation. It is real life testing or Kaggle test.csv (there is no label)

    Methods
    -------
    CrossValidation(...):
        Starts cross validated training.
    """

    def __init__(self, X, y, test_set=None):
        self.X = X
        self.y = y
        self.test_set = test_set

    def CrossValidation(self, params, fold_count=5, verbose=100, save_oofs=False, make_fold_preds=False, random_state=42 ) -> list:
        """
        Constructs and starts K-Fold Cross Validation. 

        Parameters
        ----------
            params : dict
                Hyperparameter dictionary compatible with xgboost.

            fold_count : int
                How many folds will data splitted into. (default = 5) 
            
            verbose : int
                How frequently will model print out training info. (default = 100)

            save_oofs : bool
                Is model going to make out of fold predictions. (default = False)

            make_fold_preds : bool
                Is model going to predict on the whole test set. Requires self.test_set is not none. (default = False)

            random_state : int
                Random state      

        Returns
        -------
            a list of the trained models best loss scores   
   
        """
        X_train, y_train = self.X, self.y
        
        oof_preds_xgb = np.zeros(len(X_train))
        preds = pd.DataFrame()
        score_list = []

        kf = KFold(n_splits=fold_count, shuffle=True, random_state=random_state)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_trn, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_trn, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            print(f"\nFold: {fold}")
            model = XGBRegressor(**params)
            
            model.fit(X_trn, y_trn, eval_set=[(X_val, y_val)], verbose=verbose)

            if make_fold_preds and self.test_set is not None:
                fold_prediction = model.predict(self.test_set)
                preds[f"model_{fold}_preds"] = fold_prediction
                score_list.append(model.best_score) 


            if save_oofs:
                oof_preds_xgb[val_idx] = model.predict(X_val)

        if save_oofs:
            np.save("xgb_oof_preds.npy", oof_preds_xgb)

        if make_fold_preds and self.test_set is not None:
            for i,n in enumerate(preds):
                if i == 0:
                    avg = preds[n].copy()
                else:
                    avg += preds[n]
            avg = avg * (1/int(i+1))
            
            preds["avarage"] = avg
            preds.to_csv("cv_preds.csv",index=False)

            return score_list

    def Train(self, params, do_test=True, test_size=0.2, random_state=42, verbose=100 ) -> XGBRegressor:
        """
        Starts training of a single model. 

        Parameters
        ----------
            params : dict
                Hyperparameter dictionary compatible with xgboost.

            verbose : int
                How frequently will model print out training info. (default = 100)

            random_state : int
                Random state        

            do_test : bool
                Is model going to create a test split and use it while training.
            
            test_size : float
                Size of the test split in percent wise. (default = 0.2)
        
        Returns
        -------
            The XGBoost model
        
        """
        if do_test:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
            
            model = XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=verbose)

            return model
        else: 
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            return model
       
    def OptunaSearch(self, params, trial_count=20, startup_trials=10, fold_count=3, n_jobs=1, direction="minimize") -> dict:
        """
        Starts Optuna Hyperparameter search. Aims for finding better models.  

        Parameters
        ----------
            params : dict
                Hyperparameter dictionary compatible with xgboost.

            trial_count : int
                How many trials will optuna search take (default = 20)

            startup_trials : int
                How many wide ranged startup trials will be made (default = 10)        
            
            n_jobs : int
                How many parallel process will be made. -1 for all available cores. (default = 1) 
            
            direction : string
                reducing error direction according to loss function (default = 'minimize')
        
        Returns
        -------
            best parameters as dict
        
        """ 


        from statistics import mean
        import optuna

        def objective(trial):
            return mean(self.CrossValidation(params=params,fold_count=fold_count, verbose=0))
    
        
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(
                n_startup_trials=startup_trials,  
                #multivariate=True,   
            ),

            pruner=HyperbandPruner(  
                min_resource=1,
                reduction_factor=3,
            ),

            storage='sqlite:///xgb_study.db', 
            study_name='xgb_optimization',
            load_if_exists=True,
        )
 
        
        study.optimize(
                        objective,
                        n_trials=trial_count,
                        show_progress_bar=True,
                        gc_after_trial=True,
                        n_jobs=n_jobs,
                        )
        
        return study.best_params 

    def GetXGBParams(self):
        """
        Gives information about xgboost parameters.
        """
        import json

        xgboost_regressor_params = {
            # ================= Global Configuration =================
            'verbosity': {
                'description': 'Controls verbosity of output messages',
                'default': 1,
                'options': [0, 1, 2, 3]
            },
            'nthread': {
                'description': 'Number of parallel threads',
                'default': 'Available cores',
                'range': [1, 'inf']
            },
            'use_rmm': {
                'description': 'Use RAPIDS Memory Manager (GPU only)',
                'default': False,
                'options': [True, False]
            },

            # ================ General Parameters ================
            'booster': {
                'description': 'Type of booster',
                'default': 'gbtree',
                'options': ['gbtree', 'dart', 'gblinear']
            },
            'device': {
                'description': 'Compute device',
                'default': 'cpu',
                'options': ['cpu', 'cuda', 'gpu']
            },
            'validate_parameters': {
                'description': 'Enable parameter validation',
                'default': True
            },

            # ============== Tree Booster Parameters ==============
            'eta': {
                'description': 'Learning rate',
                'default': 0.3,
                'range': [0, 1]
            },
            'gamma': {
                'description': 'Minimum loss reduction for split',
                'default': 0,
                'range': [0, 'inf']
            },
            'max_depth': {
                'description': 'Maximum tree depth',
                'default': 6,
                'range': [0, 'inf']
            },
            'sampling_method': {
                'description': 'Sampling technique',
                'default': 'uniform',
                'options': ['uniform', 'gradient_based']
            },
            'grow_policy': {
                'description': 'Tree growth strategy',
                'default': 'depthwise',
                'options': ['depthwise', 'lossguide']
            },
            'monotone_constraints': {
                'description': 'Feature monotonicity constraints',
                'default': None
            },
            'interaction_constraints': {
                'description': 'Allowed feature interactions',
                'default': None
            },

            # ============== Dart Booster Parameters ==============
            'sample_type': {
                'description': 'DART sampling method',
                'default': 'uniform',
                'options': ['uniform', 'weighted']
            },
            'rate_drop': {
                'description': 'Dropout rate for DART',
                'default': 0.0,
                'range': [0, 1]
            },
            'skip_drop': {
                'description': 'Probability to skip dropout',
                'default': 0.0,
                'range': [0, 1]
            },

            # ============ Categorical Features ============
            'max_cat_to_onehot': {
                'description': 'Threshold for one-hot encoding',
                'default': 4,
                'range': [1, 'inf']
            },
            'max_cat_threshold': {
                'description': 'Max categories per split',
                'default': 32,
                'range': [1, 'inf']
            },

            # ============== Learning Task Parameters ==============
            'objective': {
                'description': 'Regression objectives',
                'default': 'reg:squarederror',
                'options': [
                    'reg:squarederror',
                    'reg:squaredlogerror',
                    'reg:pseudohubererror',
                    'reg:quantileerror',
                    'reg:tweedie',
                    'survival:aft'
                ]
            },
            'tweedie_variance_power': {
                'description': 'Tweedie distribution parameter',
                'default': 1.5,
                'range': (1, 2)
            },
            'quantile_alpha': {
                'description': 'Quantile regression targets',
                'default': 0.5,
                'range': [0, 1]
            },
            'aft_loss_distribution': {
                'description': 'AFT survival distribution',
                'options': ['normal', 'logistic', 'extreme']
            },

            # ============== Advanced Regularization ==============
            'lambda': {
                'description': 'L2 regularization',
                'default': 1,
                'range': [0, 'inf']
            },
            'alpha': {
                'description': 'L1 regularization',
                'default': 0,
                'range': [0, 'inf']
            },
            'max_delta_step': {
                'description': 'Maximum tree output delta',
                'default': 0,
                'range': [0, 'inf']
            },

            # ============== Experimental Features ==============
            'multi_strategy': {
                'description': 'Multi-target strategy',
                'default': 'one_output_per_tree',
                'options': ['one_output_per_tree', 'multi_output_tree']
            },
            'enable_categorical': {
                'description': 'Enable categorical support',
                'default': False,
                'options': [True, False]
            }
        }
        return json.dumps(xgboost_regressor_params, indent=2, sort_keys=True)