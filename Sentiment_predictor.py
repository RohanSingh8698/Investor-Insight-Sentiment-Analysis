from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = 'results\checkpoint-500'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

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
    predicted_sentiment = sentiment_mapping[prediction]
    
    return predicted_sentiment, probabilities.cpu().numpy()

text_to_predict = "MSCI’s Asia Pacific Index - a gauge for benchmarks in the region - slipped the most in three months. Futures contracts for US shares steadied in Asian trading after the S&P 500 erased earlier gains.p"
predicted_sentiment, probabilities = predict_sentiment(text_to_predict)
print(f'Predicted Sentiment: {predicted_sentiment}')
print(f'Probabilities: {probabilities}')    
