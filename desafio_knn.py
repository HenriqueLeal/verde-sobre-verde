import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from glob import glob
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

ss = StandardScaler()

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

X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

grid_params = {'n_neighbors': [1,3,5,11,19],
               'weights': ['uniform', 'distance'],
               'metric': ['euclidean', 'manhattan']}

gs = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1,
                  cv=3, n_jobs=1)

gs_results = gs.fit(X_train, Y_train)

print(gs_results.best_score_)
print(gs_results.best_estimator_)
print(gs_results.best_params_)


knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='euclidean',
                     metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                     weights='uniform')


#CROSS VALIDATION
cv_scores_train = cross_val_score(knn, X_train, Y_train, cv=10)
cv_scores_test = cross_val_score(knn, X_test, Y_test, cv=10)
print('Cross Treino:{}'.format(np.mean(cv_scores_train)))
print('Cross Teste:{}'.format(np.mean(cv_scores_test)))
#CROSS VALIDATION

knn.fit(X_train, Y_train) 

plot_confusion_matrix(knn, X_test, Y_test)
plt.show()

print("Treino Score: {}".format(knn.score(X_train, Y_train))) #acertos treino
print("Teste Score: {}".format(knn.score(X_test, Y_test))) #acertos teste

flowerx = ss.transform(np.array([[260, 378, 496, 636, 782, 757, 994, 1207, 5787]])) #braquiaria saudavel
flower_type = classesNames[int(knn.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = ss.transform(np.array([[366, 471, 632, 796, 999,  967,  1326, 1532, 7318]])) #AMARGOSO NÃO SAUDAVEL
flower_type = classesNames[int(knn.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = ss.transform(np.array([[458, 606, 845, 1002, 1272, 1382, 1747, 2108, 8244]])) #CARURU NÃO SAUDAVEL
flower_type = classesNames[int(knn.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = ss.transform(np.array([[387, 516 ,625, 799,  947,  946,  1145, 1379, 7544]])) #JUAZEIRO NÃO SAUDAVEL
flower_type = classesNames[int(knn.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = ss.transform(np.array([[254, 347 ,439, 558,  674,  710,  846,  1007, 4331]])) #LEITEIRO NÃO SAUDAVEL
flower_type = classesNames[int(knn.predict(flowerx))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))
