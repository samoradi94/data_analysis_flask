from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from datetime import datetime
import pandas as pd
import calendar
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1
app.config['TEMPLATES_AUTO_RELOAD'] = True

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


def apply_kmeans(df, k):
    # scaler = MinMaxScaler()
    # scaler.fit(df)
    # scaler.transform(df)
    # kmeans = KMeans(n_clusters=2)
    # kmeans.fit(df.drop('Private', axis=1))

    scaler = MinMaxScaler()
    df[['scaled_recency', 'scaled_frequency', 'scaled_monetary']] = scaler.fit_transform(
        df[['recency', 'frequency', 'monetary']])

    df.drop('last_purchase_date', axis=1, inplace=True)

    kmeans = KMeans(n_clusters=k)

    df['label'] = kmeans.fit_predict(df[['scaled_recency', 'scaled_frequency', 'scaled_monetary']])
    cluster_centers = kmeans.cluster_centers_
    u_labels = list(range(0, len(cluster_centers)))

    for i in u_labels:
        plt.scatter(df[df['label'] == i]['scaled_frequency'], df[df['label'] == i]['scaled_recency'], label=i,
                    alpha=0.5, marker='o')
    plt.legend()
    plt.xlabel('freq')
    plt.ylabel('recency')

    from io import BytesIO
    import base64
    from matplotlib.figure import Figure
    # fig = Figure()
    # ax = fig.subplots()
    # for i in u_labels:
    #     ax.scatter(df[df['label'] == i]['scaled_frequency'], df[df['label'] == i]['scaled_recency'], label=i,
    #                 alpha=0.5, marker='o')
    # buf = BytesIO()
    # image_data1 = base64.b64encode(buf.getbuffer()).decode("ascii")
    # fig.savefig(buf, format="jpeg")
    if os.path.isfile('static/images/kmeans_result.jpeg'):
        os.remove('static/images/kmeans_result.jpeg')
    plt.savefig('static/images/kmeans_result.jpeg')
    plt.cla()
    return cluster_centers


def get_week_day(x):
    return x.dayofweek


def get_week_days_statistics(data):
    logging.warning(f'data {data.shape}')
    logging.warning(f'data[0] {data.loc[0]}')
    days = [[5, 'شنبه'], [6, 'یکشنبه'], [0, 'دوشنبه'], [1, 'سه شنبه'], [2, 'چهارشنبه'], [3, 'پنج شنبه'], [4, 'جمعه']]
    # days = ['Saturday', 'Sunday', 'Monday', 'Tuesday',
    #         'Wednesday', 'Thursday', 'Friday']
    results = []
    # visited_days = []
    for day in days:
        logging.warning(f'day {day}')
        logging.warning(f'unique week day {data["week_day"].unique()}')
        specific_day_data = data[(data['week_day'] == day[0])][['date', 'total_purchase']]
        logging.warning(f'specific_day_data {specific_day_data.shape}')

        counts = specific_day_data.groupby('date').count()
        # logging.warning(f'specific_day_data  {specific_day_data}')
        # logging.warning(f'counts  {counts}')
        # print(day)
        # print(counts['total_purchase'].mean())
        # print(counts['total_purchase'].std())
        # logging.warning(f"mean total purchase   {counts['total_purchase'].mean()}")
        # logging.warning(f"std total purchase  {counts['total_purchase'].std()}")
        results.append([day[1], counts['total_purchase'].mean(), counts['total_purchase'].std()])
        # visited_days.append(day)
    # logging.warning(f'day of week  {visited_days}')
    return results


def get_tsak_2_result(data):
    week_day_data = data.query(
        f'week_day != {4} and week_day != {5}')  # contains ['Monday' 'Tuesday' 'Wednesday' 'Saturday' 'Sunday']
    holiday_data = data.query(f'week_day == {4} or week_day == {5}')  # contains ['Thursday' 'Friday']

    number_of_purch_weekday = week_day_data.groupby('date').aggregate('count')
    number_of_purch_weekend = holiday_data.groupby('date').aggregate('count')

    plt.hist(number_of_purch_weekday['order_id'], bins=70, facecolor='#EB455F', edgecolor='#0D1821', alpha=0.7,
             label='workingDays')
    plt.hist(number_of_purch_weekend['order_id'], bins=50, facecolor='#FFFFD0', edgecolor='#0D1821', alpha=0.7,
             label='weekend')
    plt.ylabel('Histogram of Demand')
    plt.xlabel('Demand')
    plt.legend()
    plt.savefig('static/images/task2.jpeg')
    plt.clf()


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


# def create_figure():
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig

# @app.route('/', methods=['POST'])
# def submit_k_value():
#     return redirect(url_for('index'))

def create_cluster_centers_list(cl):
    all_group = []
    for i in range(0, len(cl)):
        group_list = []
        group_list.append(f' خوشه {i + 1}')
        for j in range(0, 3):
            group_list.append(round(cl[i][j], 3))

        all_group.append(group_list)
    return all_group


@app.route('/', methods=["GET", "POST"])
def index():
    global data
    global rfm_df

    # get data
    dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y')
    data = pd.read_csv('sample_data.csv', parse_dates=['date'], date_parser=dateparse)

    # Task 1
    data['week_day'] = data['date'].apply(get_week_day)
    logging.warning(f'day_name {data["week_day"].unique()}')

    tsk1 = get_week_days_statistics(data)
    # tsk1 = [[1, 2, 3], [1, 2, 3]]
    # Task 2
    get_tsak_2_result(data)
    task2_image = ''
    image1 = 'static/images/task2.jpeg'

    # Task 3


    if request.method == "POST":
        k = int(request.form["text"])
        logging.warning(f'K: {type(k)}')

        rfm_df = create_rfm_dataset(data)
        cluster_centers = apply_kmeans(rfm_df, k)

        centers = create_cluster_centers_list(cluster_centers)
        # logging.error(f'type: {type(cluster_c)}')
        kmean_image = 'static/images/kmeans_result.jpeg'
        return render_template('response.html', image1 = image1, tbl=tsk1, k=k, cluster_centers=centers, image2 = kmean_image)
    else:
        return render_template('index.html', image1=image1, tsk2_img=task2_image, tbl=tsk1, k=0)
    #     # index()
    #     if not k:
    #         flash('K is required!')
    #     else:
    #         rfm_df = create_rfm_dataset(data)
    #         print(k)

    # apply_kmeans(rfm_df, k)

    # return render_template('index.html', user_image=img, tsk2_img=task2_image, tbl=tsk1, k = k)


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    logging.error('Submit')
    k1 = 0
    if request.method == "POST":
        k1 = request.form['text']

    img = '1655873404431.jpeg'
    # # get data
    # dateparse = lambda x: datetime.strptime(x, '%m/%d/%Y')
    # data = pd.read_csv('sample_data.csv', parse_dates=['date'], date_parser=dateparse)

    # Task 1
    # data['week_day'] = data['date'].apply(get_week_day)
    week_statistics = get_week_days_statistics(data)

    return render_template('test.html', tbl=week_statistics, k=k1)


# @app.route('/', methods=['POST'])
# def results():
#     ...
#     # form = SearchForm(request.form)
#     if request.method == 'POST':
#         inputString = request.form.get('value_k')
#     ...
#     return render_template('index.html')

# @app.route('/', methods=('GET','POST'))
# def create():
#     if request.method == "POST":
#         k = request.form['value_k']
#
#         with app.test_request_context('/'):
#             request.method
#     return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
