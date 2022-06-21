import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

output = 'out2/'

#1 Відкрити та зчитати наданий файл з даними.
df = pd.read_csv('Shill Bidding Dataset.csv')
print("1.\nФайл 'Shill Bidding Dataset.csv' успішно прочитаний")

#2  Визначити та вивести кількість записів та кількість полів у завантаженому наборі даних.
rows, columns = df.shape
print('2.\nКількість записів: %d \nКількість полів: %d' % (rows, columns))

#3 Видалити непотрібні для аналізу атрибути та вивести перші 10 записів набору даних.
new_df = df.drop(columns=['Record_ID', 'Auction_ID', 'Bidder_ID'], axis=1)
print('3.\nПерші 10 записів(без непотрібних атрибутів):\n %s' % new_df.head(10))

#4 Розділити набір даних на навчальну (тренувальну) та тестову вибірки, попередньо перемішавши початковий набір даних.
x_train, x_test, y_train, y_test =\
train_test_split(new_df.drop(['Class'], axis=1), new_df['Class'], random_state=0)

#5 Використовуючи відповідні функції бібліотеки scikit-learn, збудувати класифікаційну модель дерева прийняття
#рішень глибини 5 та навчити її на тренувальній вибірці, вважаючи, що в наданому наборі даних цільова
#характеристика визначається останнім стовпчиком, а всі інші виступають в ролі вихідних аргументів.
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(x_train, y_train)

#6 Представити графічно побудоване дерево за допомогою бібліотеки graphviz.
classes = ['0', '1']
features = new_df.drop(['Class'], axis=1).columns
#fig, axes = plt.subplots(nrows=1,ncols=1,figsize=(6,6), dpi=300) # Дерево, побудоване за допомогою pyplot
#tree.plot_tree(clf, class_names=classes, feature_names=features)
#fig.savefig('plt_tree.png')

dot_data = tree.export_graphviz(clf,
                        feature_names=features,
                        class_names=classes,
                        filled=True)


graph = graphviz.Source(dot_data)
graph.render(filename='6 tree', directory=output, format="png", overwrite_source=True, cleanup=True)

#7 Обчислити класифікаційні метрики збудованої моделі для тренувальноїта тестової вибірки. Представити результати
#роботи моделі на тестовій вибірці графічно. Порівняти результати, отриманні при застосуванні різних критеріїв
#розщеплення: інформаційний приріст на основі ентропії чи неоднорідності Джині.

def plot_of_conf_matrix(method, ladel=''):
    clf = DecisionTreeClassifier(max_depth=5, criterion=method)
    clf.fit(x_train, y_train)

    def cms(x,y):
        predict = clf.predict(x)
        cm = confusion_matrix(y, predict, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        return disp
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    cms(x_test, y_test).plot(ax=axes[0])
    axes[0].set_title(f'Confusion matrix for test sample({method} criterion)')
    cms(x_train, y_train).plot(ax=axes[1])
    axes[1].set_title(f'Confusion matrix for train sample({method} criterion)')
    #plt.show()
    test_report = classification_report(y_test, clf.predict(x_test), digits=4)
    train_report = classification_report(y_train, clf.predict(x_train), digits=4)
    fig.savefig(f'{output}7 {method}.png')
    print(f'Метрики для тестової вибірки({method})\n{test_report}\n'
          f'Метрики для тренувальної вибірки({method}) \n{train_report} ')
print('7.')
plot_of_conf_matrix('gini')
plot_of_conf_matrix('entropy')

#8  З’ясувати вплив максимальної кількості листів та мінімальної кількості елементів у внутрішньому вузлі
#для його подальшого розбиття не результати класифікації. Результати представити графічно.

max_leaf_n = np.arange(2, 50)
min_samples_l = np.arange(1, 50)
list = []
list2 = []

for ml in max_leaf_n:
    clf1 = DecisionTreeClassifier(max_leaf_nodes=ml, random_state=0)
    clf2 = DecisionTreeClassifier(min_samples_leaf=ml, random_state=0)
    clf1.fit(x_train, y_train)
    clf2.fit(x_train, y_train)
    pred = clf1.predict(x_test)
    pred2 = clf2.predict(x_test)
    list.append(accuracy_score(y_test, pred))
    list2.append(accuracy_score(y_test, pred2))

fig, ax = plt.subplots(1,1, figsize=(15,8))
locator = ticker.MultipleLocator(base=1)
ax.xaxis.set_major_locator (locator)
plt.plot(max_leaf_n, list)
plt.plot(max_leaf_n, list2)
plt.title('8. Changes of accuracy depending on maximal leaf node and minimal sample leaf', size=16)
plt.legend(['max_leaf_nodes', 'min_sample_leaf'])
plt.xlabel('max_leaf_node/min_sample_leaf')
plt.ylabel('Accuracy')
plt.grid()
fig.savefig(f'{output}/8 graph')

#9. Навести стовпчикову діаграму важливості атрибутів, які використовувалися для класифікації
#(див. feature_importances_).Пояснити, яким чином – на Вашу думку – цю важливість можна підрахувати.

importances = pd.DataFrame({'Feature': x_train.columns, 'Importance': np.round(clf.feature_importances_, 3)})
importances = importances.sort_values('Importance', ascending=False).reset_index()

def autolabel(rects, labels=None, height_factor=1.01):
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if labels is not None:
            try:
                label = labels[i]
            except (TypeError, KeyError):
                label = ' '
        else:
            label = '%d' % round(height,3)
        ax.text(rect.get_x() + rect.get_width()/2., height_factor*height,
                '{}'.format(label),
                ha='center', va='bottom')

fig, ax = plt.subplots(1, 1, figsize=(15,8))
plt.tick_params(axis='x', which='major', labelsize=8)
plt.bar(importances['Feature'], importances['Importance'])
plt.title('9. Importance of features')
autolabel(ax.patches, round(importances['Importance'].sort_values(),3), height_factor=1.01)
fig.savefig(f'{output}9 Importance')
plt.show()

