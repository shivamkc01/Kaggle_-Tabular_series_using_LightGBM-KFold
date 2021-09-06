# Import Helpful Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

if __name__ == '__main__':
    # Load Data
    df_train = pd.read_csv("../input/train.csv", index_col='id')
    test = pd.read_csv("../input/test.csv", index_col='id')

    FEATURES = list(df_train.columns[:-1])
    TARGET = df_train.columns[-1]
    # print(df_train.head())

    # Missing Values
    df_train['n_missing'] = df_train[FEATURES].isna().sum(axis=1)
    test['n_missing'] = test[FEATURES].isna().sum(axis=1)

    df_train['std'] = df_train[FEATURES].std(axis=1)
    test['std'] = test[FEATURES].std(axis=1)

    FEATURES += ['n_missing', 'std']
    n_missing = df_train['n_missing'].copy()
    df_train[FEATURES] = df_train[FEATURES].fillna(df_train[FEATURES].mean())
    test[FEATURES] = test[FEATURES].fillna(test[FEATURES].mean())

    """
    Train Model
    Let's try out a simple XGBoost model. 
    This algorithm can handle missing values,
    but you could try imputing them instead.
    We use XGBClassifier (instead of XGBRegressor,
    for instance), since this is a classification problem.
    """

    X = df_train.loc[:, FEATURES]
    y = df_train.loc[:, TARGET]

    final_predictions = []  # list for storing our predictions values
    # Here I am creating 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X=X, y=y)):
        X_train = X.loc[train_idx]
        X_valid = X.loc[valid_idx]
        X_test = test.copy()

        y_train = y.loc[train_idx]
        y_valid = y.loc[valid_idx]

        scaler = StandardScaler()
        X_train[FEATURES] = scaler.fit_transform(X_train[FEATURES])
        X_valid[FEATURES] = scaler.transform(X_valid[FEATURES])
        X_test[FEATURES] = scaler.transform(X_test[FEATURES])

        model = LGBMClassifier(
            max_depth=4,
            num_leaves=7,
            n_estimators=10000,
            colsample_bytree=0.3,
            subsample=0.5,
            random_state=42,
            reg_alpha=18,
            reg_lambda=17,
            learning_rate=0.1,
            device='cpu',
            objective='binary'
        )

        model.fit(X_train, y_train,
                  verbose=False,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric="auc",
                  early_stopping_rounds=200)

        preds_valid = model.predict_proba(X_valid)[:, 1]
        preds_test = model.predict_proba(X_test)[:, 1]
        final_predictions.append(preds_test)
        print("-----------------------Training Model-------------------------")
        print(f"FOLD : {fold} -----> AUC : {roc_auc_score(y_valid, preds_valid)}")
        print(f"----------------------Done with {fold} fold------------------")

        # Make Submission
        """
        Our predictions are binary 0 and 1, 
        but you're allowed to submit probabilities instead.
        In scikit-learn, you would use the predict_proba method instead of predict.
        """
        preds = np.mean(np.column_stack(final_predictions), axis=1)

        y_pred = pd.Series(
            preds,
            index=X_test.index,
            name=TARGET
        )

        y_pred.to_csv("../input/Light_Gbm_preds.csv")
