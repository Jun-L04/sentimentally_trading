import pandas as pd
import json
from datetime import datetime
from pandas.tseries.offsets import BDay

# stock returns data
goog = pd.read_csv("stock_returns.csv")

print(goog.columns)

# news data
with open("test_news.json", "r") as json_file:
    google_news = json.load(json_file)
    print(len(google_news))

training_data = []

for news_item in google_news:
    headline = news_item['headline']
    news_date = news_item['datetime']

    # convert news_date to datetime object
    news_date_dt = datetime.strptime(news_date, '%Y-%m-%d')

    # next available business day for opening price
    # in case news release on weekends
    next_business_day = news_date_dt
    while next_business_day.strftime('%Y-%m-%d') not in goog['Date'].values:
        next_business_day += BDay(1)
    next_business_day_str = next_business_day.strftime('%Y-%m-%d')

    # # opening price on the next business day
    # try:
    #     open_price = goog.loc[goog['Date'] ==
    #                           next_business_day_str, 'Open'].values[0]
    # except IndexError:
    #     # exception data not found
    #     print(
    #         f"Skipping news item on {news_date} - opening price not found for {next_business_day_str}")
    #     continue

    # the next business day available after at least 3 days
    three_days_after = next_business_day + BDay(3)
    three_days_after_str = three_days_after.strftime('%Y-%m-%d')

    # aggregated 3 day return price three business days after news press
    try:
        agg_excess_return_3 = goog.loc[goog['Date'] ==
                                       three_days_after_str, '3-Day Excess Return'].values[0]
    except IndexError:
        # exception data not found
        print(
            f"Skipping news item on {news_date} - Aggregated 3 Day Excess Return Not Found on {three_days_after_str}")
        continue

    # wether the stock price went up or down
    label = 1 if agg_excess_return_3 > 0 else 0

    # make training data
    training_data.append(
        {'headline': headline, 'label': label, 'date': news_date})

# export training data
with open("training_data.json", "w") as json_file:
    json.dump(training_data, json_file, indent=4)

print(f"# of training data: {len(training_data)}")
