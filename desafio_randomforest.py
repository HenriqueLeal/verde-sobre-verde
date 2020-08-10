import pandas as pd 
from sklearn.model_selection import train_test_split 
import numpy as np
from glob import glob
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def is_healty(typePlant):
  return int(typePlant != "braquiaria")

stock_files = sorted(glob("data/_*.csv"))

all_data = pd.concat((pd.read_csv(file).assign(filename = file)
          for file in stock_files), ignore_index = True)

classesNames = [c.split('_')[-1][:-4] for c in stock_files]

all_data = all_data.drop(columns=['Function', 'Sample num'])

X = np.asarray(all_data[['F1 (410nm)', 
                         'F2 (440nm)',
                         'F3 (470nm)',
                         'F4 (510nm)',
                         'F5 (550nm)',
                         'F6 (583nm)',
                         'F7 (620nm)',
                         'F8 (670nm)',
                         'CLEAR']])
Y = np.asarray([classesNames.index(fn.split('_')[-1][:-4]) for fn in all_data['filename']])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=True) 
print("quantidade de dados de treino: {}".format(X_train.shape))
print("quantidade de dados de teste: {}".format(X_test.shape))

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

model_params = {
    'n_estimators': [50, 150, 250, 700],
    'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    'min_samples_split': [2, 4, 6]
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=model_params, cv= 5)
CV_rfc.fit(X_train, Y_train)
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

#CROSS VALIDATION
cv_scores_train = cross_val_score(CV_rfc, X_train, Y_train, cv=10)
cv_scores_test = cross_val_score(CV_rfc, X_test, Y_test, cv=10)
print('Cross Treino:{}'.format(np.mean(cv_scores_train)))
#CROSS VALIDATION

flowerx = np.array([[260, 378, 496, 636, 782, 757, 994, 1207, 5787]]) #braquiaria saudavel
flower_type = classesNames[int(CV_rfc.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = np.array([[366, 471, 632, 796, 999,  967,  1326, 1532, 7318]]) #AMARGOSO NÃO SAUDAVEL
flower_type = classesNames[int(CV_rfc.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = np.array([[458, 606, 845, 1002, 1272, 1382, 1747, 2108, 8244]]) #CARURU NÃO SAUDAVEL
flower_type = classesNames[int(CV_rfc.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = np.array([[387, 516 ,625, 799,  947,  946,  1145, 1379, 7544]]) #JUAZEIRO NÃO SAUDAVEL
flower_type = classesNames[int(CV_rfc.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = np.array([[254, 347 ,439, 558,  674,  710,  846,  1007, 4331]]) #LEITEIRO NÃO SAUDAVEL
flower_type = classesNames[int(CV_rfc.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))