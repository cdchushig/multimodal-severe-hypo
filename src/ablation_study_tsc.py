import pandas as pd
import os
import utils.consts as cons
from utils.check_patients import get_patients_id
from sklearn.model_selection import train_test_split
from trep import TRep
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from sklearn.model_selection import GridSearchCV
from utils.evaluator import compute_metrics
from sktime.classification.deep_learning import GRUClassifier


def obtain_outputs_and_metrics(model, x_test, y_test):

    y_pred_test = model.predict(x_test)

    results_test = compute_metrics(y_test, y_pred_test)

    print('Test results:', results_test)

    results_test = {f'acc': results_test['accuracy'], f'specificity': results_test['specificity'],
                    f'recall': results_test['recall'], f'roc_auc': results_test['roc_auc']}

    return  results_test


def representation_method(X_train, X_test, Y_train, Y_test, method, seed):
    estimator = RandomForestClassifier(random_state=seed, n_estimators=100, ccp_alpha=0.01)
    if method == 't-rep':
        trep = TRep(
            input_dims=1,
            time_embedding='t2v_sin',
            output_dims=20 )
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        trep.fit(X_train, n_epochs=10, verbose=1)
        X_train = trep.encode(X_train,  encoding_window='full_series')
        X_test = trep.encode(X_test, encoding_window='full_series')

    elif method == 'QUANT':
        model = QUANTClassifier(random_state=seed, estimator=estimator)
        estimator = GridSearchCV(model,
                               param_grid=cons.params_interval,
                               cv=3,
                               return_train_score=True,
                               n_jobs=9)

    elif method == 'shapelet':
        print('** Trainning RDSTClassifier**')
        model = RDSTClassifier(random_state=seed, n_jobs=9, estimator=estimator)
        estimator = GridSearchCV(model,
                               param_grid=cons.params_shapelet,
                               cv=3,
                               return_train_score=True,
                               n_jobs=9)

    elif method == 'gru':
        print('** Trainning Gru Classifier**')
        model = GRUClassifier(random_state=seed)
        estimator = GridSearchCV(model,
                                 param_grid=cons.params_gru,
                                 cv=3,
                                 return_train_score=True,
                                 n_jobs=9)

    estimator.fit(np.asarray(X_train), np.asarray(Y_train))
    results = obtain_outputs_and_metrics(estimator, np.asarray(X_test), np.asarray(Y_test))
    return results

def preprocessing(df,e):
    u = df[df['patient_id'] == e]
    u.index = pd.to_datetime(u['new'])
    u = u['mean']
    u = u.resample('10T', offset='10s').mean().interpolate(method='slinear')
    u = u.dropna()[-720:]
    initial_value = u[0]
    extension_length = 720 - len(u)
    extension = np.full(extension_length, initial_value)
    extended_array = np.concatenate((extension, u))
    return extended_array

df = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED_SIGNAL, 'Time_series_CGM.csv'))
pd1 = pd.read_csv(os.path.join(cons.PATH_PROJECT_DATA_PREPROCESSED, 'BPtRoster.txt'), sep='|')
pd1['BCaseControlStatus'] = pd1['BCaseControlStatus'].replace(['Case'], 0).replace(['Control'], 1)
pd1['label_encoded'] = pd1['BCaseControlStatus']
pd1.drop(['RecID', 'BCaseControlStatus'], axis=1, inplace=True)
patients = get_patients_id()
data = patients.merge(pd1, on=['PtID'])

for method in ['gru','shapelet','QUANT','t-rep']:
    results_final=pd.DataFrame()
    for seed in cons.SEEDS:
        X_train, X_test, Y_train, Y_test = train_test_split(data['PtID'], data['label_encoded'],
                                                            test_size=0.2,
                                                            random_state=seed,
                                                            stratify=data['label_encoded'])

        x_train =[]
        for e in X_train:
            signal =preprocessing(df, e)
            x_train.append(signal)

        x_test =[]
        for e in X_test:
            signal =preprocessing(df, e)
            x_test.append(signal)

        results =representation_method(x_train, x_test, Y_train, Y_test, method, seed)
        df_results = pd.DataFrame([results])
        results_final = pd.concat([results_final, df_results])
    mean_df = results_final.mean()
    results_final = results_final.append(mean_df)
    results_final.to_csv(os.path.join(cons.PATH_PROJECT_REPORTS), f'results_2_{method}.csv')



