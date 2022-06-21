import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#1 Відкрити та зчитати файл з даними:
df = pd.read_csv('Weather.csv')

#2 Визначити та вивести кількість записів та кількість полів у кожному записі:
count_of_rows, count_of_columns = df.shape
print('Кількість записів: %d \nКількість полів: %d' % (count_of_rows, count_of_columns))

#3 Вивести 5 записів, починаючи з М-ого (число М – місяць народження студента, має бути визначено як змінна), та кожен
# N-ий запис, де число N визначається як 500 * М для місяця з першого півріччя та 300 * М для місяця з другого півріччя:
M = 8 #august
N = 300 * M #second half of the year
first_5_lines = df[M:M+5]
every_300th_line = df[::300]
print('Перші 5 рядків, починаючи з 8: \n%s \nКожен 300-ий запис: \n%s ' % (first_5_lines, every_300th_line))

#4 Визначити та вивести тип полів кожного запису:
type_of_columns = df.dtypes
print('Типи полів: \n%s' % type_of_columns)

#5 Замість поля СЕТ ввести нові текстові поля, що відповідають числу, місяцю та року
# Місяць та число повинні бути записані у двоцифровому форматі

df.insert(0, 'Day', pd.DatetimeIndex(df['CET']).day)
df.insert(1, 'Month', pd.DatetimeIndex(df['CET']).month)
df.insert(2, 'Year', pd.DatetimeIndex(df['CET']).year)
new_df = df.drop(['CET'], axis=1)
#print(new_df)

#6 Визначити та вивести:
#a кількість днів із порожнім значенням поля Events:
days_with_empty_events = len(new_df[new_df[' Events'].isnull() == True])
print('Кількість днів з порожнім значенням Events: ', days_with_empty_events)

#b день, у який середня вологість була мінімальною, а також швидкості вітру в цей день:
min_humidity = min(new_df[' Mean Humidity'])
day = new_df[new_df[' Mean Humidity'] == min_humidity][['Year', 'Month', 'Day', ' Max Wind SpeedKm/h', ' Mean Wind SpeedKm/h']]
print('День з найменьшою середньою вологістю та швидкості вітру в цей день: \n%s' % day)

#c місяці, коли середня температура від нуля до п’яти градусів:
#aaa = new_df.groupby(['Month'], as_index=False).agg({'Mean TemperatureC': 'mean'})
#print(aaa)
months_with_temp_btw_0_and_5 = new_df[new_df['Mean TemperatureC'].between(0,5)]['Month'].unique()
print('Місяці з середньою температурою від 0 до 5 градусів: \n%s' % np.sort(months_with_temp_btw_0_and_5))

#7 Визначити та вивести:
#a Середню максимальну температуру по кожному дню за всі роки;
mean_max_temp = new_df.groupby(['Year'])['Max TemperatureC'].mean()
print('Середня максимальня температура по кожному дню за всі роки: \n%s' % round(mean_max_temp, 2))

# Варіант 2
ss = new_df.groupby(['Day', 'Year']).agg({'Max TemperatureC': 'mean'}).reset_index()
ss = ss.rename(columns={'Max TemperatureC': 'Mean max Temperature'})
#print('Середня максимальня температура по кожному дню за всі роки: \n%s' % round(ss, 2))

#b Кількість днів у кожному році з туманом.
sorted_by_event = new_df[new_df[' Events'].str.contains('Fog') == True]
count_fog = (sorted_by_event.groupby(['Year'], as_index=False)[' Events'].count())
count_fog = count_fog.rename(columns={' Events': ' Days with Fog'})
print('Перелік кількості днів з туманом для кожного року: \n%s' % count_fog)

#8 Побудувати стовпчикову діаграму кількості Events:
group_by_events = new_df.groupby([' Events'], dropna=True).size()
fig1 = plt.figure()
group_by_events.plot.bar(width=0.95, color='b')
plt.title('Event bar', size=18)
plt.xlabel('Event', size=16)
plt.ylabel('Count of event', size=16)

#9 Побудувати кругову діаграму напрямків вітру (сектор на діаграмі має відповідати одному з восьми напрямків – північний, південний, східний,західний та проміжні):
new_df['Dir'] = new_df['WindDirDegrees'].apply(lambda x: 'Northern' if -23 < x < 23 else 'Northeastern' if x < 68 else 'Eastern' if x < 113 else
                                                         'Southeastern' if x < 158 else 'Southern' if x < 203 else 'Southwestern' if x < 248 else
                                                         'Western' if x < 293 else 'Northwestern' if x < 338 else 'Northern')
group_by_Dir = new_df.groupby(['Dir']).count().reset_index()
fig2 = plt.figure()
plt.pie(group_by_Dir['Day'], labels=group_by_Dir['Day'], radius=1, autopct='%1.1f%%', wedgeprops={'lw':1, 'ls':'--','edgecolor':"k"}, startangle=54, rotatelabels=True)
plt.legend(group_by_Dir['Dir'],fontsize='x-small')

#10 Побудувати на одному графіку (тип графіка обрати самостійно!):
#a Середню по кожному місяцю кожного року максимальну температуру
graf1 = new_df.groupby(by=['Year', 'Month'], dropna=False, as_index=False)[['Max TemperatureC']].mean().round(3)

#b Середню по кожному місяцю кожного року мінімальну точку роси
graf2 = new_df.groupby(by=['Year', 'Month'], dropna=False, as_index=False)[['Min DewpointC']].mean().round(3)
graf2['Month/Year'] = graf2['Month'].astype(str)+ '/' + graf2['Year'].astype(str)
fig3 = plt.figure()
plt.plot(graf2['Month/Year'],graf1['Max TemperatureC'],  color='green', label='Average maximal temperature')
plt.plot(graf2['Month/Year'],graf2['Min DewpointC'], 'r', label='Average minimal dewpoint')
plt.ylabel('Temperature, C', size=18)
plt.xlabel('Month/Year', size=18)
plt.xticks(fontsize=5)
plt.legend()
plt.grid()
#plt.show()