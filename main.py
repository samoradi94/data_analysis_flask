import pandas as pd
from datetime import datetime
import calendar
import matplotlib.pyplot as plt

def get_week_day(x):
    return calendar.day_name[x.weekday()]


dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y')
data = pd.read_csv('sample_data.csv', parse_dates=['date'], date_parser=dateparse)

data['week_day'] = data['date'].apply(get_week_day)

# eda
plt.bar(data[data['week_day'] == 'Monday'], height=1)
plt.show()

plt.hist(data['total_purchase'], bins=1000)
plt.show()

print(data[data['week_day'] == 'Monday']['total_purchase'].max())
print(data[data['week_day'] == 'Monday']['total_purchase'].min())
mean = data['total_purchase'].groupby(data['week_day']).agg('mean')
std = data['total_purchase'].groupby(data['week_day']).agg('std')

print(mean)
print(std)
print(data['total_purchase'].groupby(data['week_day']).aggregate('std'))