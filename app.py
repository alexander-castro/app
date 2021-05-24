import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline

DIR = os.path.dirname(os.path.realpath('__file__'))
app = Flask(__name__)
CORS(app)
categorical_columns = ['male', 'education', 'currentSmoker', 'prevalentStroke', 'prevalentHyp', 'diabetes']
numeric_columns = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

column_trans_1 = make_column_transformer((MinMaxScaler(), numeric_columns), remainder='passthrough')
column_trans_2 = make_column_transformer((OrdinalEncoder(dtype=np.int32), categorical_columns), remainder='passthrough')

with app.app_context():
    data = pd.read_csv(os.path.join(DIR, '..//data/framingham_heart_disease.csv'), sep=';')
    data = data.dropna()
    data = data.drop(data.columns[0], axis=1)
    data['BMI'] = data['BMI'].apply(lambda x: x.replace(',','.'))
    data['sysBP'] = data['sysBP'].apply(lambda x: x.replace(',','.'))
    data['diaBP'] = data['diaBP'].apply(lambda x: x.replace(',','.'))

    X = data.drop("TenYearCHD", axis=1)
    y = data["TenYearCHD"]
    y = pd.DataFrame(LabelEncoder().fit_transform(y))

    X = pd.DataFrame(column_trans_2.fit_transform(X), columns=np.concatenate((categorical_columns, numeric_columns), axis=None), index=X.index)
    X = pd.DataFrame(column_trans_1.fit_transform(X), columns=np.concatenate((numeric_columns, categorical_columns), axis=None), index=X.index)
    resampler = RandomUnderSampler(random_state=1234, sampling_strategy=0.7)
    X, y = resampler.fit_resample(X, y)

    ct = ColumnTransformer([
        ('num', MinMaxScaler(),
        make_column_selector(dtype_include=np.number)),
        ('cat',
        OrdinalEncoder(dtype=np.int32),
        make_column_selector(dtype_include=object))], remainder='passthrough')

    estimators = [
                ('lr', LogisticRegression(max_iter=10000, random_state=0, class_weight='balanced', solver='liblinear', penalty='l2'))
                ]
    pipe = Pipeline(estimators)
    clf = pipe.fit(X, y.values.ravel())
    dump(clf, os.path.join(DIR,'..//model/dataModel.joblib'))

@app.route('/api/patients', methods=['POST'])
def postPatients():
    model = load(os.path.join(DIR,'..//model/dataModel.joblib'))
    cols = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']
    X = pd.DataFrame(data=request.json).transpose()
    X = pd.DataFrame(X, columns=cols)
    X = pd.DataFrame(column_trans_2.transform(X), columns=np.concatenate((categorical_columns, numeric_columns), axis=None), index=X.index)
    X = pd.DataFrame(column_trans_1.transform(X))
    X.columns=np.concatenate((numeric_columns, categorical_columns), axis=None)
    result = {}
    result['response'] = model.predict(X.iloc[[0]]).tolist()[0]
    result['probability'] = model.predict_proba(X.iloc[[0]]).tolist()[0][1]
    return jsonify({"Result": result})

if __name__ == '__main__':
    app.run(threaded=True, port=5000)
    