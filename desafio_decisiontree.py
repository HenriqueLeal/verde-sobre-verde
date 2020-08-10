import pandas as pd 
from sklearn.model_selection import train_test_split 
import numpy as np
from glob import glob
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0) 
print("quantidade de dados de treino: {}".format(X_train.shape))
print("quantidade de dados de teste: {}".format(X_test.shape))

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}

clf = GridSearchCV(DecisionTreeClassifier(), parameters)
clf.fit(X_train, Y_train)
clf = clf.fit(X_train, Y_train)

cross_score_treino = cross_val_score(clf, X_train, Y_train, cv=5)
cross_score_teste = cross_val_score(clf, X_test, Y_test, cv=5)
print('Cross Treino:{}'.format(np.mean(cross_score_treino)))
print('Cross Teste:{}'.format(np.mean(cross_score_teste)))

print(clf.score(X_train, Y_train))
print(clf.score(X_test, Y_test))

flowerx = np.array([[260, 378, 496, 636, 782, 757, 994, 1207, 5787]]) #braquiaria saudavel
flower_type = classesNames[int(clf.predict(ss.transform(flowerx)))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = np.array([[366, 471, 632, 796, 999,  967,  1326, 1532, 7318]]) #AMARGOSO NÃO SAUDAVEL
flower_type = classesNames[int(clf.predict(ss.transform(flowerx)))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = np.array([[458, 606, 845, 1002, 1272, 1382, 1747, 2108, 8244]]) #CARURU NÃO SAUDAVEL
flower_type = classesNames[int(clf.predict(ss.transform(flowerx)))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = np.array([[387, 516 ,625, 799,  947,  946,  1145, 1379, 7544]]) #JUAZEIRO NÃO SAUDAVEL
flower_type = classesNames[int(clf.predict(ss.transform(flowerx)))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))

flowerx = np.array([[254, 347 ,439, 558,  674,  710,  846,  1007, 4331]]) #LEITEIRO NÃO SAUDAVEL
flower_type = classesNames[int(clf.predict(ss.transform(flowerx)))]
print("Saida = %d, flor da classe: %s" % (is_healty(flower_type),flower_type))