# [Хакатон Гринатом](https://hackathon.greenatom.ru/)
19 нобря 2019 года  

Трек "Машинное обучение" одна из трех команд финалистов (без места)

Цель - предсказать веротяность увольнения сотрудника.

Анализ, обучение модели и постороение графа [EDA_fit_graph.ipynb](https://github.com/polosataya/grinatom-hakaton/blob/master/EDA_fit_graph.ipynb)  
Обученные модели tfidf.pkl и lgb_clf.txt в [pythonanywhere](https://github.com/polosataya/grinatom-hakaton/tree/master/pythonanywhere)

Предсказание [predict.py](https://github.com/polosataya/grinatom-hakaton/blob/master/predict.py). 
Два варианта - заполнение данных в консоли и загрузка из файла test_data.csv

Работающее приложение по адресу http://polosataya.pythonanywhere.com/  
Для http://pythonanywhere.com/ выполнить   
```pip3.7 install joblib lightgbm --user```  
Код в папке [pythonanywhere](https://github.com/polosataya/grinatom-hakaton/tree/master/pythonanywhere)  
