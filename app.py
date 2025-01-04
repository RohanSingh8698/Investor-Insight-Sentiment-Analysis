from flask import Flask, request, render_template
from newsapi import NewsApiClient
import datetime
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from collections import Counter

app = Flask(__name__)
newsapi = NewsApiClient(api_key='16e57a18d676454d8c38f14c634273f7')
model_path = 'results\checkpoint-500'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).cpu().numpy()[0]
    sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return sentiment_mapping[prediction], probabilities.cpu().numpy()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'summarize' in request.form:
            text_to_summarize = request.form['text_to_summarize']
            summary = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)
            summary_text = summary[0]['summary_text']
            sentiment, _ = predict_sentiment(summary_text)
            return render_template('result.html', summary=summary_text, sentiment=sentiment)

        category_input = request.form.get('category').lower()
        valid_categories = ['finance', 'trade', 'investment']
        if category_input not in valid_categories:
            return render_template('index.html', error="Invalid category. Please choose a valid category.")

        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=15)
        articles_page = newsapi.get_everything(q=category_input,
                                               from_param=start_date.strftime('%Y-%m-%d'),
                                               to=end_date.strftime('%Y-%m-%d'),
                                               language='en',
                                               sort_by='relevancy',
                                               page=1)
        articles = articles_page['articles']
        titles = [article['title'] for article in articles]
        sentiments = [predict_sentiment(title)[0] for title in titles]
        majority_class = Counter(sentiments).most_common(1)[0][0]
        
        return render_template('result.html', majority_class=majority_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)





