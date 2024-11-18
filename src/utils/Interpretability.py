import os
import utils.consts as consts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.classifiers import train_compute_metrics
from utils.consts import SEEDS
import shap
from PyALE import ale
import math
import dice_ml
from matplotlib.lines import Line2D

continuous_features = ['medications3_2', 'medications3_0', 'cde', 'dee', 'edd', 'hgg', 'fgf', 'bbc', 'dcc', 'bcd',
'GrPegNonTotTime', 'TrailMakBTotTime', 'GrPegDomTotTime', 'TrailMakATotTime', 'FrailtyFirstWalkTotTimeSec',
'SymbDigOTotCorr','SymbDigWTotCorr', 'BGVisit2', 'DukeSocSatScore', 'FrailtySecWalkTotTimeSec']

def call_clf_interpretability_SHAP(train_databases, test_databases,
                clfs, features=[],SHAP='Kernel',seeds=36,path=consts.PATH_PROJECT_FUSION_FIGURES):

    if SHAP== 'Kernel':
        list_SHAP = train_interpretability_SHAP(clfs, train_databases, test_databases, features,SHAP=SHAP)
        list2=[]
        for e in list_SHAP:
            list2.extend(e)

        i=pd.concat([test_databases[0].drop('label',axis=1),test_databases[1].drop('label',axis=1),test_databases[2].drop('label',axis=1),test_databases[3].drop('label',axis=1),test_databases[4].drop('label',axis=1)])
        dataframe = pd.DataFrame(list2, columns=train_databases[0].columns[:-1])
        absolute_sums = dataframe.abs()
        result =  pd.DataFrame({key: absolute_sums[cols].sum(axis=1) for key, cols in consts.dict_names.items()})
        importances =result.sum().to_dict()
        plt.close()
        shap.summary_plot(np.asarray(list2),i.astype('float'),max_display=len(dataframe.columns))
        plt.savefig(os.path.join(path, 'SHAP_Early_bee.pdf'))



        # o=abs(dataframe).mean()
        # o1 = abs(dataframe).std()
        # df = pd.DataFrame(columns=['mean', 'std'])
        # df['mean'] = o.values
        # df['std'] = o1.values
        # df.index = o.index
        # df=df.sort_values(by=['mean'],ascending=False)
        # fig, ax = plt.subplots()
        # ax.barh(df.index, xerr=df['std'],width=df['mean'],capsize=3)
        # ax.invert_yaxis()
        # fig.tight_layout()
        # plt.show()
        # plt.savefig(os.path.join(path, 'SHAP_Early.pdf'))
        return  dataframe
    elif SHAP== 'decision_plot'or SHAP == 'waterfall' or SHAP== 'force_plot':

        x_train = train_databases.drop(['label'], axis=1)
        x_test = test_databases.drop(['label'], axis=1)
        y_train = train_databases['label']
        y_test = test_databases['label']

        if len(features) == 0:

            SHAP_values = train_compute_metrics(clfs, x_train, y_train,
                                                x_test, y_test, seeds, SHAP=SHAP)
        else:
            SHAP_values = train_compute_metrics(clfs, x_train[features], y_train, x_test[features], y_test,
                                                seed=seeds, SHAP=SHAP)


def plot_final_importance(dictionary):
    plt.figure(figsize=(8, 6))
    groups = list(dictionary.keys())
    values = list(dictionary.values())
    colors = ['skyblue', 'salmon', 'red', 'green', 'grey']

    x_positions = [5, 2, 8, 2, 8]   # Valores aleatorios entre 0 y 10 para el eje X
    y_positions =  [5, 2, 8, 8, 2 ] # Valores aleatorios entre 0 y 10 para el eje Y
    # Dibujar los círculos
    for i, (group, value) in enumerate(dictionary.items()):
        plt.scatter(x_positions[i], y_positions[i], s=value * 200, color=colors[i], alpha=0.6)
        if i == 4 :
            plt.text(x_positions[i], y_positions[i], str(np.round(value)), color='black', ha='center', va='center', fontsize=14)
        else:
            plt.text(x_positions[i], y_positions[i], str(np.round(value)), color='black', ha='center', va='center', fontsize=14)
    # Personalizar gráfico
    plt.xticks([])  # Quitar las etiquetas del eje X
    plt.yticks([])  # Quitar las etiquetas del eje Y
    plt.xlim([0,10])
    plt.ylim([0, 10])
    # for spine in plt.gca().spines.values():
    #     spine.set_visible(False)
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=15, label='Unaware'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=15, label='Fear'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], markersize=15, label='Totscores'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3], markersize=15, label='CGM'),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[4], markersize=15, label='Medications')]
    plt.legend(handles=legend_handles,title="Datasets", loc='upper right', bbox_to_anchor=(1.22, 1), borderaxespad=0.)
    plt.savefig(os.path.join(consts.PATH_PROJECT_FUSION_FIGURES, 'Importance_datasets.pdf'), dpi=300, bbox_inches='tight')

    plt.figure(figsize=(8, 6))
    pd.Series(dictionary).plot(kind='bar', x='Categoría', y='Valor', color='skyblue', legend=False)
    plt.xticks(rotation=0, fontsize=13)
    # Agregar título y etiquetas
    plt.xlabel('Dataset', fontsize=15)
    plt.ylabel('Importance Value', fontsize=15)
    # Mostrar el gráfico
    plt.savefig(os.path.join(consts.PATH_PROJECT_FUSION_FIGURES, 'Importance_datasets_bar.pdf'), dpi=300,
                bbox_inches='tight')

    # Mostrar gráfico
    # plt.show()
def call_cf_interpretability_cnf(train_databases, test_databases,
                clfs, path=consts.PATH_PROJECT_FUSION_FIGURES):

    df_cnf = train_interpretability_cnf(clfs, train_databases, test_databases)

    plt.figure(figsize=(10, 6))  # Tamaño de la figura

    plt.scatter(df_cnf[df_cnf['value'] > 0]['value'], df_cnf[df_cnf['value'] > 0]['Feature'], color='blue', )

    # Puntos negativos en rojo
    plt.scatter(df_cnf[df_cnf['value'] < 0]['value'], df_cnf[df_cnf['value'] < 0]['Feature'], color='red',)

    plt.xlabel('Variations', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Counterfactuals', fontsize=16)
    plt.grid(axis='y')


    plt.savefig(os.path.join(path, 'Counterfactuals.pdf'), dpi=300, bbox_inches='tight')


def train_interpretability_cnf(clf_name, train_databases, test_databases):
    pandas_differences= pd.DataFrame(columns=['Feature', 'value'])

    for idx in range(len(train_databases)):
        df_train = train_databases[idx].astype(float)
        df_test = test_databases[idx].astype(float)


        features = list(df_train.drop('label', axis=1).columns.values)
        data_dice = dice_ml.Data(dataframe=df_train,
                                # # For perturbation strategy
                                continuous_features=continuous_features,
                                outcome_name='label')

        model = train_compute_metrics(clf_name, df_train.drop('label', axis=1), df_train['label'],
                                      df_test.drop('label', axis=1), df_test['label'], seed=SEEDS[idx], obtain_model=True)

        model_dice = dice_ml.Model(model=model,
                                   # There best_clf backends for tf, torch, ...
                                   backend="sklearn")

        explainer = dice_ml.Dice(data_dice,
                                 model_dice,
                                 # Random sampling, genetic algorithm, kd-tree,...
                                 method="random")

        # Create explanation
        # Generate CF based on the blackbox model
        df_test_no_label = df_test.drop('label', axis=1)
        pred = model.predict(df_test_no_label)

        for samples in range(len(df_test)):
            if df_test.label[samples] == 1.0 and pred[samples] == 1.0:
                input_datapoint = df_test.iloc[samples:samples+1]
                input_datapoint_aux = input_datapoint.drop('label', axis=1)


                print(input_datapoint)
                cf = explainer.generate_counterfactuals(input_datapoint_aux,
                                                        total_CFs=5,
                                                        desired_class="opposite")

                # Visualize it
                print('2')
                cf.visualize_as_dataframe(show_only_changes=True)
                new_dataset = cf.cf_examples_list[0].final_cfs_df

                comparison = new_dataset.reset_index(drop=True) !=  input_datapoint.reset_index(drop=True)

                diff_columns = comparison.any()
                different_columns = diff_columns[diff_columns].index.tolist()
                different_columns.remove('label')
                for e in different_columns:
                    val_dif = input_datapoint[e].values - new_dataset[e].values
                    df_aux = pd.DataFrame([e,val_dif[0]])
                    df_aux= df_aux.T
                    df_aux.columns = ['Feature', 'value']
                    pandas_differences = pd.concat([pandas_differences,df_aux])
    return pandas_differences





def train_interpretability_SHAP(clf_name, train_databases, test_databases, features, SHAP=False):
    SHAP_list=[]
    if SHAP == 'Kernel':
        for i, j in enumerate(train_databases):
            x_train = j.drop(['label'], axis=1)
            x_test = test_databases[i].drop(['label'], axis=1)
            y_train = j['label']
            y_test = test_databases[i]['label']

            if len(features) == 0:
                SHAP_values = train_compute_metrics(clf_name, x_train, y_train,
                                                    x_test, y_test, SEEDS[i], SHAP=SHAP)
            else:
                SHAP_values = train_compute_metrics(clf_name, x_train[features], y_train, x_test[features], y_test,
                                                    seed=SEEDS[i],SHAP=SHAP)
            SHAP_list.append(SHAP_values)
        return SHAP_list


def interpretability_ALE(train,test,clf_name,Partition_seed ,features=[],numerical_features=[],categorical_features=[], continuous_features=[]):
    x_train = train.drop(['label'], axis=1)
    x_test = test.drop(['label'], axis=1)
    y_train = train['label']
    y_test = test['label']
    if len(features) == 0:
        model = train_compute_metrics(clf_name, x_train, y_train,
                                            x_test, y_test, seed=Partition_seed,ALE= True)
    else:
        model = train_compute_metrics(clf_name, x_train[features], y_train, x_test[features], y_test,
                                            seed=Partition_seed, ALE=True)
    n, c = get_categorical_numerical_names(x_train)
    for e in x_train.columns[0:15]:
        if e in n:
            ale_eff = ale(
            X=x_test, model=model, feature=[e], grid_size=50, include_CI=True, C=0.95, feature_type='discrete')
        elif e in c:
            ale_eff = ale(
                X=x_test, model=model, feature=[e], grid_size=50, include_CI=True, C=0.95, feature_type='continuous')



def get_categorical_numerical_names(df_data: pd.DataFrame) -> (list, list):

    df_info = identify_type_features(df_data)
    list_numerical_vars = list(df_info[df_info['type'] == 'c'].index)
    list_categorical_vars = list(df_info[df_info['type'] == 'd'].index)

    return list_categorical_vars, list_numerical_vars


def identify_type_features(df, discrete_threshold=10, debug=False):
    """
    Categorize every feature/column of df as discrete or continuous according to whether or not the unique responses
    are numeric and, if they are numeric, whether or not there are fewer unique renponses than discrete_threshold.
    Return a dataframe with a row for each feature and columns for the type, the count of unique responses, and the
    count of string, number or null/nan responses.
    """
    counts = []
    string_counts = []
    float_counts = []
    null_counts = []
    types = []
    for col in df.columns:
        responses = df[col].unique()
        counts.append(len(responses))
        string_count, float_count, null_count = 0, 0, 0
        for value in responses:
            try:
                val = float(value)
                if not math.isnan(val):
                    float_count += 1
                else:
                    null_count += 1
            except:
                try:
                    val = str(value)
                    string_count += 1
                except:
                    print('Error: Unexpected value', value, 'for feature', col)

        string_counts.append(string_count)
        float_counts.append(float_count)
        null_counts.append(null_count)
        types.append('d' if len(responses) < discrete_threshold or string_count > 0 else 'c')

    feature_info = pd.DataFrame({'count': counts,
                                 'string_count': string_counts,
                                 'float_count': float_counts,
                                 'null_count': null_counts,
                                 'type': types}, index=df.columns)
    if debug:
        print(f'Counted {sum(feature_info["type"] == "d")} discrete features and {sum(feature_info["type"] == "c")} continuous features')

    return feature_info







