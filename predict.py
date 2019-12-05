#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:37:44 2019

@author: tanya
"""

import numpy as np
import pandas as pd
import joblib
#import pickle
import lightgbm as lgb
#import catboost
#from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
#from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import date

#Ввод данных из консоли
print("Группы должностей, которые вы можете выбрать: 'Бухгалтер' 'Инженер' 'Офисный сотрудник' 'Руководитель' \
      'ИТ' 'Обслуживающий персонал' 'Рабочий' 'Продавец' 'Консультант' 'Оператор call-центра'")

position = input("Выберите наиболее похожую на вашу должность: ")
years = input("Сколько лет вы проработали на этом месте? '>10', '5-10', '3-5', '1-2', '<1' ")

salary = input("Оцените уровень оплаты труда 1 2 3 4 5: ")
superiors = input("Оцените начальство 1 2 3 4 5: ")
workplace = input("Оцените рабочее место 1 2 3 4 5: ")
colleagues = input("Оцените коллектив 1 2 3 4 5: ")
career = input("Оцените карьерный рост 1 2 3 4 5: ")

plus = input("Опишите плюсы работы: ")
minus = input("Что можно улучшить: ")

data = {'Дата отзыва': [str(date.today())], 'Должность': [position], 'Лет': [years], 'Оплата труда': [salary], 
        'Начальство': [superiors], 'Рабочее место': [workplace], 'Коллектив': [colleagues], 
        'Карьерный рост': [career], 'Плюсы': [plus], 'Минусы': [minus],  } #'Статус': [s],
data = pd.DataFrame(data=data)

#Загрузка из файла
#col_list = ['Дата отзыва', 'Должность', 'Лет', 'Оплата труда', 'Начальство', 'Рабочее место', 'Коллектив', 
#            'Карьерный рост', 'Плюсы', 'Минусы'] #, 'Статус'
#data = pd.read_csv("test_data.csv")[col_list]

data['Дата отзыва'] = pd.to_datetime(data['Дата отзыва'])
data['year'] = 2000 - data['Дата отзыва'].dt.year
data[['Оплата труда', 'Начальство', 'Рабочее место', 'Коллектив', 
      'Карьерный рост']] = data[['Оплата труда', 'Начальство', 'Рабочее место', 'Коллектив', 
                                 'Карьерный рост']].fillna(0).astype('int8')

#Схлопываем множество должностей в 10 основных
buh_list = ['Бухгалтер']
ingen_list = ['Инженер']
ofis_list = ['Офисный сотрудник']
meneg_list = ['Руководитель']
it_list = ['ИТ']
obsl_list = ['Обслуживающий персонал']
work_list = ['Рабочий']
trade_list = ['Продавец']
cons_list = ['Консультант']
op_list = ['Оператор call-центра']

data['Бухгалтер'] = np.where(data['Должность'].isin(buh_list), 1, 0)
data['Инженер'] = np.where(data['Должность'].isin(ingen_list), 1, 0)
data['Офисный сотрудник'] = np.where(data['Должность'].isin(ofis_list), 1, 0)
data['Руководитель'] = np.where(data['Должность'].isin(meneg_list), 1, 0)
data['ИТ'] = np.where(data['Должность'].isin(it_list), 1, 0)
data['Обслуживающий персонал'] = np.where(data['Должность'].isin(obsl_list), 1, 0)
data['Рабочий'] = np.where(data['Должность'].isin(work_list), 1, 0)
data['Продавец'] = np.where(data['Должность'].isin(trade_list), 1, 0)
data['Консультант'] = np.where(data['Должность'].isin(cons_list), 1, 0)
data['Оператор call-центра'] = np.where(data['Должность'].isin(op_list), 1, 0)

data['Лет'] = data['Лет'].map({'>10': 5, '5-10': 4, '3-5': 3, '1-2': 2, '<1': 1,}).fillna(0).astype('int8')

#Объединим отзыв в одно текстовое поле
data['Отзыв'] = data['Плюсы'] + ' ' + data['Минусы']

features =['Лет', 'Оплата труда', 'Начальство', 'Рабочее место', 'Коллектив', 'Карьерный рост', 'Бухгалтер', 
           'Инженер', 'Офисный сотрудник', 'Руководитель', 'Обслуживающий персонал', 'Рабочий', 'Продавец', 
           'Консультант', 'Оператор call-центра', 'year',] 

tfidf = TfidfVectorizer(min_df=3,ngram_range=(1, 5),  ) 
# Загружаем модель
tfidf = joblib.load('tfidf.pkl') 
X_sparse = tfidf.transform(data['Отзыв'])

X_hstack = csr_matrix(hstack([data[features], X_sparse]))

cat_clf = CatBoostClassifier() #task_type='GPU'
cat_clf.load_model('cat_clf.cbm')
cat_predict = cat_clf.predict(X_hstack, prediction_type='Probability')[:, 1]

lgb_clf = lgb.Booster(model_file = 'lgb_clf.txt')
lgb_predict = lgb_clf.predict(X_hstack, num_iteration=lgb_clf.best_iteration) 

print('Вероятность вашего увольнения:', 100*cat_predict[0].round(4), '%')
print('Вероятность вашего увольнения:', 100*lgb_predict[0].round(4), '%')
