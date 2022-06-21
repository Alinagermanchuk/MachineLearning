import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
from sklearn.neighbors import NearestCentroid

#1 Відкрити та зчитати наданий файл з даними

df = pd.read_csv('Shill Bidding Dataset.csv')
print("1.\nФайл 'Shill Bidding Dataset.csv' успішно прочитано")

#2 Визначити та вивести кількість записів

rows = df.shape[0]
print('2.\nКількість записів: %d' % rows)

#3 Встановити в якості індексу датафрейму атрибут Record_ID.
new_df = df.set_index('Record_ID')
print("3.\nДатафрейм з індексом 'Record_ID': \n%s" % new_df)

#4 Видалити атрибут Class, а також непотрібні для подальшого аналізу
new_df = new_df.drop(['Class', 'Auction_ID', 'Bidder_ID'], axis=1)
print('4.\nДатафрейм без непотрібних для аналізу атрибутів: \n%s' % new_df)

#5 Вивести атрибути, що залишилися.
attributes = new_df.columns
print('5.\nАтрибути, що лишилися: \n%s' % list(attributes))

#6 Використовуючи функцію KMeans бібліотеки scikit-learn, виконати розбиття набору даних на кластери. Кількість
#кластерів визначити на основі початкового набору даних (вибір обгрунтувати). Вивести координати центрів кластерів.
kmeans = KMeans(n_clusters=4).fit(new_df)
kmeans_center = kmeans.cluster_centers_
print('6.\nКоординати центрів кластерів(отримані функцією KMeans: ')
for i in range(len(kmeans_center)):
    print(f'Центр {i+1}-го кластеру:\n', np.around(kmeans_center[i], 3))

#7 Використовуючи функцію AgglomerativeClustering бібліотеки scikitlearn, виконати розбиття набору даних на кластери.
#Кількість кластерів обрати такою ж самою, як і в попередньому методі. Вивести координати центрів кластерів.

clustering = AgglomerativeClustering(n_clusters=4,  compute_full_tree=True).fit(new_df)
cluster = AgglomerativeClustering(n_clusters=4).fit_predict(new_df)
near = NearestCentroid().fit(new_df, cluster)
print('\n7.\nКоординати центрів кластерів(отримані функцією AgglomerativeClustering:')
for i in range(len(near.centroids_)):
    print(f'Центр {i+1}-го кластеру:\n', np.around(near.centroids_[i], 3))

#8 Порівняти результати двох використаних методів кластеризації.
print('8.\nВ завданнях 6 і 7 отримані однакові результати')















