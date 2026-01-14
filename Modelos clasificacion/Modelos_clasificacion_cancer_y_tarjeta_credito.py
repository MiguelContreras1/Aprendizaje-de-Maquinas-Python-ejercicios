import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
    roc_curve
)

from skopt import BayesSearchCV
from skopt.space import Real, Integer

from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
import aequitas.plot as ap
from aequitas.plotting import Plot

df_cancer = pd.read_csv(
    './breast-cancer-wisconsin.data',
    names=[
        'id',
        'clump_thcknss',
        'unif_cell_sz',
        'unif_cell_shp',
        'marg_adhesion',
        'epith_cell_sz',
        'bare_nuclei',
        'blnd_chromatin',
        'normal_nucleoli',
        'mitoses',
        'class'
    ],
    na_values='?'
)

# Recodificar variable dependiente
df_cancer['class'] = np.where(df_cancer['class'] == 2, 0, 1)

# Excluir variable id para ciertos análisis
df_cancer_data = df_cancer.drop('id', axis=1)

# Histogramas por variable
df_melt = df_cancer.melt('id', var_name='variable', value_name='value')
sns.catplot(data=df_melt, x='value', row='variable',
            kind='count', sharex=False, sharey=False)

# Estadísticos descriptivos
stats = df_cancer_data.describe()

nan_counts = df_cancer_data.isna().sum()
nan_counts.name = 'nan_counts'

stats = stats.append(nan_counts)

# Correlaciones de las variables con respecto a la variable dependiente
df_cancer_data.corr()['class']

# Gráfico de las variables con respecto a la variable dependiente
df_melt2 = df_cancer_data.melt(
    'class', var_name='variable', value_name='value')
sns.catplot(data=df_melt2, hue='class', x='value', y='variable',
            kind='bar', ci=None, orient='h')

# Preprocesamiento de datos

X = df_cancer_data.drop('class', axis=1)
y = df_cancer_data['class']

# Semilla para los procesos que involucran aleatoriedad
seed = 10

# División entrenamiento y prueba
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=seed
)


def build_pipe(step):
    # Estandarizar e imputar missing values
    pipeline_base = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    # Añadir clasificador según necesidad
    return Pipeline(pipeline_base.steps + [step])


# Creación de pipelines para cada clasificador
pipe_knn = build_pipe(('knn', KNeighborsClassifier()))
pipe_tree = build_pipe(('tree', DecisionTreeClassifier(random_state=seed)))
pipe_rf = build_pipe(('rf', RandomForestClassifier(random_state=seed)))
pipe_ridge = build_pipe(
    ('ridge', LogisticRegression(penalty='l2'))
)
pipe_lasso = build_pipe(
    ('lasso', LogisticRegression(penalty='l1', solver='liblinear'))
)
pipe_logit = build_pipe(('logit', LogisticRegression(penalty='none')))

# Creación de espacios de búsqueda de hiperparámetros para cada clasificador
space_knn = {'knn__n_neighbors': Integer(1, 15, prior='uniform')}
space_tree = {
    'tree__min_samples_leaf': Integer(1, 20, prior='uniform'),
    'tree__max_depth': Integer(1, 15, prior='uniform')
}
space_rf = {
    'rf__min_samples_leaf': Integer(1, 20, prior='uniform'),
    'rf__max_depth': Integer(1, 15, prior='uniform'),
    'rf__n_estimators': Integer(1, 2000, prior='uniform')
}
space_ridge = {'ridge__C': Real(1e-10, 1e+1, prior='log-uniform')}
space_lasso = {'lasso__C': Real(1e-10, 1e+1, prior='log-uniform')}

# Creación de métrica a maximizar en la búsqueda de hiperparámetros
# (media armónica entre accuracy, precision, recall y AUC)


def custom_score(y_actual, y_pred):
    acc = accuracy_score(y_actual, y_pred)
    pre = precision_score(y_actual, y_pred)
    rec = recall_score(y_actual, y_pred)
    auc = roc_auc_score(y_actual, y_pred)
    return (4/((1/acc)+(1/pre)+(1/rec)+(1/auc)))

# Búsqueda bayesiana de hiperparámetros con base en la métrica definida


def build_search(pipe, space):
    search = BayesSearchCV(
        pipe,
        space,
        n_iter=50,
        cv=5,
        n_jobs=-1,
        verbose=3,
        random_state=10,
        scoring=make_scorer(custom_score)
    )
    return search


search_knn = build_search(pipe_knn, space_knn)
search_tree = build_search(pipe_tree, space_tree)
search_rf = build_search(pipe_rf, space_rf)
search_ridge = build_search(pipe_ridge, space_ridge)
search_lasso = build_search(pipe_lasso, space_lasso)

# Ajustar modelos a la muestra de entrenamiento
search_knn.fit(X_train, y_train)
search_tree.fit(X_train, y_train)
search_rf.fit(X_train, y_train)
search_ridge.fit(X_train, y_train)
search_lasso.fit(X_train, y_train)
pipe_logit.fit(X_train, y_train)

# Obtener accuracy, precision, recall y AUC para los modelos


def get_metrics(pipe, search=None, logit=False):
    if not logit:
        pipe.set_params(**search.best_params_)
        pipe.fit(X_test, y_test)
    acc = accuracy_score(pipe.predict(X_test), y_test)
    pre = precision_score(pipe.predict(X_test), y_test)
    rec = recall_score(pipe.predict(X_test), y_test)
    auc = roc_auc_score(pipe.predict(X_test), y_test)
    fpr, tpr, _ = roc_curve(pipe.predict(X_test), y_test)
    # Crear curva ROC
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return {'acc': acc, 'pre': pre, 'rec': rec, 'auc': auc}


metrics_knn = get_metrics(pipe_knn, search_knn)
metrics_tree = get_metrics(pipe_tree, search_tree)
metrics_rf = get_metrics(pipe_rf, search_rf)
metrics_ridge = get_metrics(pipe_ridge, search_ridge)
metrics_lasso = get_metrics(pipe_lasso, search_lasso)
metrics_logit = get_metrics(pipe_logit, logit=True)


# Modelo de clasificación para tarjetas de crédito
df_credit = pd.read_csv('./credit_card.csv', index_col=0)

# Recodificar variables binarias y crear dummies para variable de raza
dummies = pd.get_dummies(
    df_credit[['owner', 'selfemp', 'sex', 'race']], drop_first=True
)
df_credit = pd.concat(
    [
        df_credit,
        dummies
    ],
    axis=1
)

# Preprocesamiento de datos
X2 = df_credit.drop(['card', 'owner', 'selfemp', 'sex', 'race'], axis=1)
y2 = df_credit['card']


def get_df(clf):
    # Predecir probabilidades
    probs = clf.predict_proba(X2)
    # Encontrar punto de corte óptimo en la curva ROC
    fpr, tpr, thresholds = roc_curve(y2, probs[:, 1], drop_intermediate=False)
    opt_threshold = thresholds[np.argmax(tpr - fpr)]
    # Hacer predicciones de acuerdo con el punto de corte óptimo
    preds = np.where(probs[:, 1] > opt_threshold, 1, 0)
    preds = pd.Series(preds, name='preds')
    # Unir columna de predicciones al resto del data frame
    df_preds = pd.concat([preds, df_credit], axis=1).drop(
        dummies.columns, axis=1)
    df_preds.rename(columns={'preds': 'score',
                    'card': 'label_value'}, inplace=True)
    # Seleccionar variables de interés para el análisis de sesgo
    df_preds_cat = df_preds[['score', 'label_value', 'sex', 'race']]
    return df_preds_cat


def get_groups(df):
    # Calcular métricas para las variables de interés
    g = Group()
    xtab, _ = g.get_crosstabs(df)
    # Definir grupos de referencia para medir el sesgo
    b = Bias()
    bdf = b.get_disparity_predefined_groups(
        xtab,
        original_df=df,
        ref_groups_dict={
            'race': 'White',
            'sex': 'Male'
        }
    )
    return bdf


# Definir métricas de interés y nivel de tolerancia
metrics = ['fnr', 'for', 'ppr']
tau = 0.8
disparity_tolerance = 1/tau

# Ajustar modelo 1 y analizar sesgo
classifier = LogisticRegression(penalty='none')
classifier.fit(X2, y2)
df1 = get_df(classifier)
bdf1 = get_groups(df1)
# Gráfico de disparidad en raza
ap.disparity(bdf1, metrics, 'race', fairness_threshold=disparity_tolerance)
# Gráfico de disparidad en sexo
ap.disparity(bdf1, metrics, 'sex', fairness_threshold=disparity_tolerance)


# Ajustar modelo 2  y analizar sesgo
calibrated_clf = CalibratedClassifierCV(
    LogisticRegression(penalty='none'),
    method='sigmoid',
    cv=5,
    n_jobs=-1
)
calibrated_clf.fit(X2, y2)
df2 = get_df(calibrated_clf)
bdf2 = get_groups(df2)
# Gráfico de disparidad en raza
ap.disparity(bdf2, metrics, 'race', fairness_threshold=disparity_tolerance)
# Gráfico de disparidad en sexo
ap.disparity(bdf2, metrics, 'sex', fairness_threshold=disparity_tolerance)

# Ajustar modelo 3 y analizar sesgo
calibrated_clf2 = CalibratedClassifierCV(
    LogisticRegression(penalty='none'),
    method='isotonic',
    cv=5,
    n_jobs=-1
)
calibrated_clf2.fit(X2, y2)
df3 = get_df(calibrated_clf2)
bdf3 = get_groups(df3)
# Gráfico de disparidad en raza
ap.disparity(bdf3, metrics, 'race', fairness_threshold=disparity_tolerance)
# Gráfico de disparidad en sexo
ap.disparity(bdf3, metrics, 'sex', fairness_threshold=disparity_tolerance)
