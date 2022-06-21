import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


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


#5 Отримати двадцять варіантів перемішування набору даних та розділення його на навчальну (тренувальну) та тестову
#вибірки, використовуючи функцію ShuffleSplit. Сформувати начальну та тестові вибірки на основі обраного користувачем
#варіанту З’ясувати збалансованість набору даних

attributes = new_df.drop(columns=['Class'], axis=1)
answers = new_df['Class']
rs = ShuffleSplit(n_splits=20, test_size=0.25, random_state=0)
split = rs.split(attributes, answers)
list_of_splits = list(split)
print('5.')

for iter, indexes in enumerate(list_of_splits, start=1):
    print(f'Індекси тренувальної/тестової вибірок №{iter}:  {indexes[0]} / {indexes[1]}')


while True:
    try:
        inp = int(input('Введіть № варіанту перемішування набору: '))
        x_train, y_train = attributes.iloc[list_of_splits[inp-1][0]], answers.iloc[list_of_splits[inp-1][0]]
        x_test, y_test = attributes.iloc[list_of_splits[inp-1][1]], answers.iloc[list_of_splits[inp-1][1]]

        break
    except (ValueError, IndexError):
        print('Введіть ЧИСЛО від 0 до 20')
inp = 5

x_train, y_train = attributes.iloc[list_of_splits[inp][0]], answers.iloc[list_of_splits[inp][0]]
x_test, y_test = attributes.iloc[list_of_splits[inp][1]], answers.iloc[list_of_splits[inp][1]]

#6Використовуючи функцію KNeighborsClassifier бібліотеки scikit-learn, збудувати класифікаційну модель на основі методу
#k найближчих сусідів (значення всіх параметрів залишити за замовчуванням) та навчити її на тренувальній вибірці,
#вважаючи, що цільова характеристика визначається стовпчиком Class, а всі інші виступають в ролі вихідних аргументів.

knc = KNeighborsClassifier()
knc.fit(x_train, y_train)

#7 Обчислити класифікаційні метрики збудованої моделі для тренувальної та тестової вибірки. Представити результати
# роботи моделі на тестовій вибірці графічно


cm = confusion_matrix(y_test, knc.predict(x_test), labels=knc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knc.classes_)
fig, ax = plt.subplots(1, 1)
disp.plot(ax=ax)
ax.set_title(f'7. Матриця невідповідностей для тестової вибірки \n(Модель побудована на основі методу k найближчих сусідів)', size=14)
test_report = classification_report(y_test, knc.predict(x_test), digits=4)
train_report = classification_report(y_train, knc.predict(x_train), digits=4)

print(f'7.\nМетрики для тестової вибірки: \n{test_report}\nМетрики для тренувальної вибірки: \n{train_report} ')
a = input('Для продовження натисніть будь-який символ: ')
plt.show()

#Обрати алгоритм KDTree та з’ясувати вплив розміру листа (від 20 до 80 з кроком 2) на результати класифікації.
#Результати представити графічно

start, end, step = 20, 80, 2

iter = range(start, end, step)

points_for_graph = []
for i in iter:
    knc_kd_tree = KNeighborsRegressor(algorithm='kd_tree', leaf_size=i)
    knc_kd_tree.fit(x_train, y_train)
    prediction_knc_kd_tree = knc_kd_tree.predict(x_test)

    accuracy = accuracy_score(y_test, prediction_knc_kd_tree.astype(int))
    points_for_graph.append(accuracy)

fig, ax = plt.subplots(1, 1)
plt.plot(iter, points_for_graph)
plt.title('8. Вплив розміру листа на результати класифікації', size=14)
plt.xlabel('Leaf node')
plt.ylabel('Accuracy')

b = input('Для продовження натисніть будь-який символ: ')
plt.show()



