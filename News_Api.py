from newsapi import NewsApiClient
import datetime
import pandas as pd
newsapi = NewsApiClient(api_key='16e57a18d676454d8c38f14c634273f7')
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=29)
print("Please choose a category from the following: 'finance', 'trade', 'investment'")
category_input = input("Enter your choice: ").lower()

valid_categories = ['finance', 'trade', 'investment']
if category_input not in valid_categories:
    print("Invalid category choice. Please run the script again with a valid category.")
    exit()

all_headlines = []
for i in range(1):
    articles_page = newsapi.get_everything(q=category_input,
                                           from_param=start_date.strftime('%Y-%m-%d'),
                                           to=end_date.strftime('%Y-%m-%d'),
                                           language='en',
                                           sort_by='relevancy',
                                           page=i+1)
    articles = articles_page['articles']
    all_headlines.extend(articles)

df = pd.DataFrame(all_headlines)
csv_filename = "news_headlines.csv"
df = df[['title']] 
df['label'] = ''  
df.to_csv(csv_filename, index=False)
print(f"Headlines saved to {csv_filename}")



# 16e57a18d676454d8c38f14c634273f7


