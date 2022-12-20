import pandas as pd
from datetime import datetime
import calendar
import matplotlib.pyplot as plt

def get_week_day(x):
    return calendar.day_name[x.weekday()]


dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y')
data = pd.read_csv('sample_data.csv', parse_dates=['date'], date_parser=dateparse)

data['week_day'] = data['date'].apply(get_week_day)
print(data.head())

# eda
# plt.bar(data[data['week_day'] == 'Monday'], height=1)
# plt.show()

# plt.hist(data['total_purchase'], bins=1000)
# plt.show()


# Task 1
print(data[data['week_day'] == 'Monday']['total_purchase'].max())
print(data[data['week_day'] == 'Monday']['total_purchase'].min())
mean = data['total_purchase'].groupby(data['week_day']).agg('mean')
std = data['total_purchase'].groupby(data['week_day']).agg('std')

print(mean)
print(std)
print(data['total_purchase'].groupby(data['week_day']).aggregate('std'))

# Task 2
week_day_data = data.query('week_day != "Thursday" and week_day != "Friday"') # contains ['Monday' 'Tuesday' 'Wednesday' 'Saturday' 'Sunday']
holiday_data = data.query('week_day == "Thursday" or week_day == "Friday"') # contains ['Thursday' 'Friday']

number_of_purch_weekday = week_day_data.groupby('date').aggregate('count')
number_of_purch_weekend = holiday_data.groupby('date').aggregate('count')


plt.hist(number_of_purch_weekday['order_id'], bins= 70,  facecolor='#B4CDED', edgecolor='#0D1821', alpha=0.7, label='workingDays')
plt.hist(number_of_purch_weekend['order_id'], bins= 30, facecolor='#8C8FE3', edgecolor='#0D1821', alpha=0.7, label='weekend')
plt.ylabel('Histogram of Demand')
plt.xlabel('Demand')
plt.legend()
plt.show()

# Task 3
