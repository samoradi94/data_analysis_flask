import pandas as pd
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')  # ignore warnings


def get_week_day(x):
    return calendar.day_name[x.weekday()]


def get_week_days_statistics():
    days = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    for day in days:
        specific_day_data = data[data['week_day'] == day][['date', 'total_purchase']]
        counts = specific_day_data.groupby('date').count()

        print(day)
        print(counts.mean())
        print(counts.std())


def calculate_recency(x):
    return (datetime.now().date() - x.date()).days


def create_rfm_dataset(data):
    rfm_df = pd.DataFrame()

    # R

    rfm_df['last_purchase_date'] = data['date'].groupby(data['user_id']).aggregate('max')
    rfm_df['recency'] = rfm_df['last_purchase_date'].apply(calculate_recency)

    # F
    rfm_df['frequency'] = data['order_id'].groupby(data['user_id']).aggregate('count')

    # M
    rfm_df['monetary'] = data['total_purchase'].groupby(data['user_id']).aggregate('sum')

    return rfm_df


def apply_kmeans(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaler.transform(df)

def find_outliers_IQR(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    IQR = q3 - q1
    outliers = df[((df < (q1 - 1.5 * IQR)) | (df > (q3 + 1.5 * IQR)))]
    return outliers


dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y')
data = pd.read_csv('sample_data.csv', parse_dates=['date'], date_parser=dateparse)

data['week_day'] = data['date'].apply(get_week_day)

# check for duplicates
dup = data[data.duplicated()]
if len(dup) == 0:
    # print('no duplicates...')
    pass
else:
    data.drop_duplicates(keep='first')

# throw unnecessary columns out
data = data.drop(['latitude', 'longitude'], axis=1)

# eda
print(data.info())
print(data['total_purchase'].describe())

# data cleaning
mis_val = data.isnull().sum()
mis_val_percent = 100 * data.isnull().sum() / len(data)
# print(mis_val)
# print(f'percentage of missing values: {mis_val_percent}')

data.dropna(axis=0, inplace=True)


# visualization
plt.hist(data['total_purchase'], bins = 50)
plt.show()

# outlier detection

sns.boxplot(data['total_purchase'])
plt.show()

# investigate an outlier point
# print(data[data['total_purchase'] == 100000000])
# print(data[data['user_id'] == 57937])

# outliers = find_outliers_IQR(data['total_purchase'])
# print('number of outliers: '+ str(len(outliers)))
# print('max outlier value: '+ str(outliers.max()))
# print('min outlier value: '+ str(outliers.min()))


# plt.bar(data[data['week_day'] == 'Monday'], height=1)
# plt.show()

# plt.hist(data['total_purchase'], bins=1000)
# plt.show()




# Task 1
get_week_days_statistics()

# Task 2
week_day_data = data.query(
    'week_day != "Thursday" and week_day != "Friday"')  # contains ['Monday' 'Tuesday' 'Wednesday' 'Saturday' 'Sunday']
holiday_data = data.query('week_day == "Thursday" or week_day == "Friday"')  # contains ['Thursday' 'Friday']

number_of_purch_weekday = week_day_data.groupby('date').aggregate('count')
number_of_purch_weekend = holiday_data.groupby('date').aggregate('count')

plt.hist(number_of_purch_weekday['order_id'], bins=70, facecolor='#B4CDED', edgecolor='#0D1821', alpha=0.7,
         label='workingDays')
plt.hist(number_of_purch_weekend['order_id'], bins=30, facecolor='#8C8FE3', edgecolor='#0D1821', alpha=0.7,
         label='weekend')
plt.ylabel('Histogram of Demand')
plt.xlabel('Demand')
plt.legend()
plt.show()

# Task 3
rfm_df = create_rfm_dataset(data)
apply_kmeans(rfm_df)