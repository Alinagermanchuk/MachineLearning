import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, log_loss
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit

#1 Відкрити та зчитати наданий файл з даними
df = pd.read_csv('Shill Bidding Dataset.csv')
print("1.\nФайл 'Shill Bidding Dataset.csv' успішно прочитаний")

#2 Визначити та вивести кількість записів та кількість полів у завантаженому наборі даних
rows, columns = df.shape
types_of_columns = df.dtypes
print('2.\nКількість записів: %d \nКількість полів: %d' % (rows, columns))
print('Типи полів:\n%s' % types_of_columns)

#3 Вивести перші 10 записів набору даних
first_ten = df.head(10)
print('3.\nПерші 10 записів:\n %s' % first_ten)

#4 Видалити непотрібні для подальшого аналізу атрибути
new_df = df.drop(columns=['Record_ID', 'Auction_ID', 'Bidder_ID'], axis=1)

#5 Отримати п’ять варіантів перемішування набору даних та розділення його на навчальну (тренувальну) та тестову
#вибірки, використовуючи функцію ShuffleSplit. Сформувати начальну та тестові вибірки на основі четвертого варіанту
attributes = new_df.drop(columns=['Class'], axis=1)
answers = new_df['Class']
rs = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
split = rs.split(attributes, answers)
list_of_splits = list(split)
print('5.')
for iter, indexes in enumerate(list_of_splits, start=1):
    print(f'Індекси тренувальної/тестової вибірок №{iter}:  {indexes[0]} / {indexes[1]}')

x_train, y_train = attributes.iloc[list_of_splits[3][0]], answers.iloc[list_of_splits[3][0]]
x_test, y_test = attributes.iloc[list_of_splits[3][1]], answers.iloc[list_of_splits[3][1]]

#6 Використовуючи відповідні функції бібліотеки scikit-learn, збудувати класифікаційну модель логістичної регресії
#та навчити її на тренувальній вибірці, вважаючи, що цільова характеристикавизначається стовпчиком Class, а всі
#інші виступають в ролі вихідних аргументів
lm = linear_model.LogisticRegression(random_state=0)
lm.fit(x_train, y_train)

print(lm.predict(x_test))
print(y_test)

#7 Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки.
#Представити результати роботи моделі на тестовій вибірці графічно.
def pred(x, constr=lm):
    return constr.predict(x)


def cms(x, y, label=lm):
    cm = confusion_matrix(y, pred(x, constr=label), labels=label.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lm.classes_)
    return disp


fig, ax = plt.subplots(1, 1)
cms(x_test, y_test).plot(ax=ax)
ax.set_title(f'7. Матриця невідповідностей для тренувальної вибірки \n(logistic regression criterion)', size=14)
test_report = classification_report(y_test, pred(x_test), digits=4)
train_report = classification_report(y_train, pred(x_train), digits=4)
print(f'7.\nМетрики для тестової вибірки: \n{test_report}\nМетрики для тренувальної вибірки: \n{train_report} ')

#8З’ясувати вплив параметру регуляризації(від 0 до 10 000) на результати класифікації. Результати представити графічно.
lim = np.linspace(0.0001, 1, 500)
accur = []
for i in lim:
    slm = linear_model.LogisticRegression(random_state=0, C=1 / i).fit(x_train, y_train)
    accuracy = accuracy_score(y_test, pred(x_test, constr=slm))
    accur.append(accuracy)
fig, ax = plt.subplots(1, 1)
plt.plot(lim, accur)
plt.title('8. Вплив параметру регуляризації на результати класифікації', size=14)
plt.xlabel('Inverse Regularization')
plt.ylabel('Accuracy')

#9 Проаналізувати ступінь впливу атрибутів на результат класифікації. Збудувати класифікаційну модель логістичної
#регресії, залишивши від 3 до 5 найбільш важливих атрибутів та порівняти її результати із моделлю з п. 7.

importances = pd.DataFrame({'Feature': x_train.columns, 'Importance': lm.coef_[0]})
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
            label = '%d' % round(height, 3)
        ax.text(rect.get_x() + rect.get_width() / 2., height_factor * height,
                '{}'.format(label),
                ha='center', va='bottom')


locator = ticker.MultipleLocator(base=0.5)
fig, ax = plt.subplots(1, 1)
ax.yaxis.set_major_locator(locator)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.title('9. Важливість атрибутів')
plt.xlabel('Features', size=14)
plt.ylabel('Importance', size=14)
plt.bar(importances['Feature'], importances['Importance'])
autolabel(ax.patches, round(importances['Importance'].sort_values(), 3), height_factor=1.01)

#plt.show()


fig, axes = plt.subplots(2, 2)
plt.subplots_adjust(hspace=0.5)
print('9.')
x = [1, 1, 0]
y = [1, 0, 1]

nums = np.arange(3, 6)
for i in nums:
    reduced_attributes = new_df[importances.iloc[:i, 1].values]
    rx_train, ry_train = reduced_attributes.iloc[list_of_splits[3][0]], answers.iloc[list_of_splits[3][0]]
    rx_test, ry_test = reduced_attributes.iloc[list_of_splits[3][1]], answers.iloc[list_of_splits[3][1]]
    rlm = linear_model.LogisticRegression(random_state=0)
    rlm.fit(rx_train, ry_train)

    cms(rx_test, ry_test, label=rlm).plot(ax=axes[x[i - 3], y[i - 3]])
    axes[x[i - 3], y[i - 3]].set_title(
        f'Матриця невідповідностей для тренувальної вибірки \n(Кількість аргументів зменшена до {i})', size=10)
    train_report = classification_report(ry_test, pred(rx_test, constr=rlm), digits=4)
    print(
        f'Кількість аргументів - {i}\nМетрики для тестової вибірки: \n{test_report}\nМетрики для тренувальної вибірки: \n{train_report} ')

cms(x_test, y_test).plot(ax=axes[0, 0])
axes[0, 0].set_title(f'Матриця невідповідностей для тренувальної вибірки \n(Звичайна кількість аргуменів)', size=10)
test_report = classification_report(y_test, pred(x_test), digits=4)

plt.show()



