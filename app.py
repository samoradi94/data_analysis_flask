from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pandas as pd
import calendar
import matplotlib.pyplot as plt


app = Flask(__name__)


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
    # scaler = MinMaxScaler()
    # scaler.fit(df)
    # scaler.transform(df)
    ...


def get_week_day(x):
    return calendar.day_name[x.weekday()]


def get_week_days_statistics(data):
    days = [['Saturday', 'شنبه'], ['Sunday', 'یکشنبه'], ['Monday', 'دوشنبه'], ['Tuesday', 'سه شنبه'],
            ['Wednesday', 'چهارشنبه'], ['Thursday', 'پنج شنبه'], ['Friday', 'جمعه']]
    results = []
    for day in days:
        specific_day_data = data[data['week_day'] == day[0]][['date', 'total_purchase']]
        counts = specific_day_data.groupby('date').count()

        # print(day)
        # print(counts['total_purchase'].mean())
        # print(counts['total_purchase'].std())
        results.append([day[1], counts['total_purchase'].mean(), counts['total_purchase'].std()])
    return results


def get_tsak_2_result(data):
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
    plt.savefig('static/images/task2.jpeg')


# def create_figure():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig


@app.route('/', methods=["GET", "POST"])
def index():
    img = '1655873404431.jpeg'
    # get data
    dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y')
    data = pd.read_csv('sample_data.csv', parse_dates=['date'], date_parser=dateparse)

    # Task 1
    data['week_day'] = data['date'].apply(get_week_day)
    tsk1 = get_week_days_statistics(data)

    # Task 2
    get_tsak_2_result(data)
    task2_image = 'task2.jpeg'

    # Task 3
    k = 0
    if request.method == "POST":
        k = request.form.get('value_k')

    rfm_df = create_rfm_dataset(data)
    apply_kmeans(rfm_df)
    return render_template('main_page.html', user_image=img, tsk2_img=task2_image, tbl=tsk1, n_k=k)


if __name__ == '__main__':
    app.run()
