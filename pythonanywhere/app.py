#!flask/bin/python
# -*- coding: utf-8 -*-
from flask import Flask
from flask import render_template
from flask import request

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
#from catboost import CatBoostClassifier
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method=='POST':

        #result = request.form['position'] # ok
        #result = request.form.position # error

        data = {'Дата отзыва': [str(date.today())],
                'Должность': [request.form['position']],
                'Лет': [request.form['years']],
                'Оплата труда': [request.form['salary']],
                'Начальство': [request.form['superiors']],
                'Рабочее место': [request.form['workplace']],
                'Коллектив': [request.form['colleagues']],
                'Карьерный рост': [request.form['career']],
                'Плюсы': [request.form['plus']],
                'Минусы': [request.form['minus']],  }

        data = pd.DataFrame(data=data)

        data['Дата отзыва'] = pd.to_datetime(data['Дата отзыва'])
        data['year'] = 2000 - data['Дата отзыва'].dt.year
        data[['Оплата труда', 'Начальство', 'Рабочее место', 'Коллектив',
              'Карьерный рост']] = data[['Оплата труда', 'Начальство', 'Рабочее место', 'Коллектив',
                               'Карьерный рост']].replace('Оценить', 0).fillna(0).astype('int8')

        #Схлопываем множество должностей в 10 основных
        buh_list = ['Бухгалтер']
        ingen_list = ['Инженер']
        ofis_list = ['Офисный сотрудник']
        meneg_list = ['Руководитель']
        it_list = ['ИТ специалист']
        obsl_list = ['Обслуживающий персонал']
        work_list = ['Рабочий']
        trade_list = ['Продавец']
        cons_list = ['Консультант']
        op_list = ['Оператор call-центра']

        data['Бухгалтер'] = np.where(data['Должность'].isin(buh_list), 1, 0)
        data['Инженер'] = np.where(data['Должность'].isin(ingen_list), 1, 0)
        data['Офисный сотрудник'] = np.where(data['Должность'].isin(ofis_list), 1, 0)
        data['Руководитель'] = np.where(data['Должность'].isin(meneg_list), 1, 0)
        data['ИТ специалист'] = np.where(data['Должность'].isin(it_list), 1, 0)
        data['Обслуживающий персонал'] = np.where(data['Должность'].isin(obsl_list), 1, 0)
        data['Рабочий'] = np.where(data['Должность'].isin(work_list), 1, 0)
        data['Продавец'] = np.where(data['Должность'].isin(trade_list), 1, 0)
        data['Консультант'] = np.where(data['Должность'].isin(cons_list), 1, 0)
        data['Оператор call-центра'] = np.where(data['Должность'].isin(op_list), 1, 0)

        data['Лет'] = data['Лет'].map({'>10': 5, '5-10': 4, '3-5': 3, '1-2': 2, '<1': 1,}).fillna(0).astype('int8')

        #Объединим отзыв в одно текстовое поле
        data['Отзыв'] = data['Плюсы'] + ' ' + data['Минусы']
        data['Count_plus'] = data['Плюсы'].str.count(" ") + 1
        data['Count_minus'] = data['Минусы'].str.count(" ") + 1

        features =['year', 'Лет', 'Оплата труда', 'Начальство', 'Рабочее место', 'Коллектив', 'Карьерный рост',
                   'Бухгалтер', 'Инженер', 'Офисный сотрудник', 'Руководитель', 'ИТ специалист',
                   'Обслуживающий персонал', 'Рабочий', 'Продавец', 'Консультант', 'Оператор call-центра',
                   'Count_plus', 'Count_minus',]

        #Стоп слова
        stop = ['не', 'на', 'что', 'за', 'по', 'все', 'как', 'это', 'то', 'но', 'только', 'так', 'для', 'если', 'очень',
                'из', 'от', 'ты', 'бы', 'вы', 'или', 'меня', 'еще', 'их', 'же', 'мне', 'до', 'да', 'без', 'при', 'там',
                'уже', 'они', 'кто', 'когда', 'здесь', 'будет', 'всех', 'чем', 'со', 'всегда', 'сейчас', 'вас', 'где',
                'быть', 'во', 'чтобы', 'мы', 'тоже', 'хотя', 'надо',]

        tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 5), stop_words=stop)
        # load your model
        try:
            tfidf = joblib.load('/home/polosataya/mysite/tfidf.pkl')
        except Exception as e:
            result = str(e)
        else:

            X_sparse = tfidf.transform(data['Отзыв'])

            X_hstack = csr_matrix(hstack([data[features], X_sparse]))

            #cat_clf = CatBoostClassifier() #task_type='GPU'
            #cat_clf.load_model('cat_clf.cbm')

            #cat_predict = cat_clf.predict(X_hstack, prediction_type='Probability')[:, 1]

            #result = (100 * cat_predict[0]).round(2)

            lgb_clf = lgb.Booster(model_file = '/home/polosataya/mysite/lgb_clf.txt')
            lgb_predict = lgb_clf.predict(X_hstack, num_iteration=lgb_clf.best_iteration)

            result = (100 * lgb_predict[0]).round(2)

    return render_template('info.html', title='Home', result=result)
