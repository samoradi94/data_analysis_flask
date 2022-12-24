from datetime import datetime
from flask import Flask, render_template, request
import logging
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


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

    if os.path.isfile('static/images/kmeans_result.jpeg'):
        os.remove('static/images/kmeans_result.jpeg')
    plt.savefig('static/images/kmeans_result.jpeg')
    plt.cla()
    return cluster_centers


def get_week_day(x):
    return x.dayofweek


def get_week_days_statistics(data):
    days = [[5, 'شنبه'], [6, 'یکشنبه'], [0, 'دوشنبه'], [1, 'سه شنبه'], [2, 'چهارشنبه'], [3, 'پنج شنبه'], [4, 'جمعه']]
    results = []
    for day in days:
        specific_day_data = data[(data['week_day'] == day[0])][['date', 'total_purchase']]
        counts = specific_day_data.groupby('date').count()
        results.append([day[1], counts['total_purchase'].mean(), counts['total_purchase'].std()])

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

    # get data
    date_parse = lambda x: datetime.strptime(x, '%m/%d/%Y')
    data = pd.read_csv('sample_data.csv', parse_dates=['date'], date_parser=date_parse)

    # remove rows with missing values
    data.dropna(axis=0, inplace=True)

    if not os.path.exists('static/images'):
        os.mkdir('static/images')

    # Task 1
    data['week_day'] = data['date'].apply(get_week_day)
    task_1 = get_week_days_statistics(data)

    # Task 2
    get_tsak_2_result(data)
    task2_image = 'static/images/task2.jpeg'

    # Task 3
    if request.method == "POST":
        k = int(request.form["text"])

        rfm_df = create_rfm_dataset(data)
        cluster_centers = apply_kmeans(rfm_df, k)

        centers = create_cluster_centers_list(cluster_centers)

        kmean_image = 'static/images/kmeans_result.jpeg'
        return render_template('response.html', image1=task2_image, tbl=task_1, k=k, cluster_centers=centers,
                               image2=kmean_image)
    else:
        return render_template('index.html', image1=task2_image, tbl=task_1)


if __name__ == '__main__':
    app.run(debug=True)
