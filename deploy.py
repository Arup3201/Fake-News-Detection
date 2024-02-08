""""
Streamlit app for fake news detection.
Developed by Arup Jana.
"""

import streamlit as st
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import joblib
from PIL import Image

nltk.download('english')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Funtion for cleaning the news text
def clean_news(news):
    cleaned_news = news.replace("â€", "").replace("â€œ", "").replace("â€˜", "").replace("â€™", "")
    cleaned_news = news.lower()
    cleaned_news = re.sub(r'httpS+', '', cleaned_news)
    cleaned_news = re.sub(r'bit.ly/S+', '', cleaned_news)

    return cleaned_news

# Count number of characters in the text
def count_characters(text):
    return len(text)

# Count number of words in the sentence
def count_words(text):
    words = text.split()
    return len(words)

# Count number of capital characters
def count_capchars(text):
    count = 0
    for word in text:
        if word.isupper():
            count+=1

    return count

# Count capital words in the text
def count_capwords(text):
    cap_words = map(str.isupper, text.split())
    return sum(cap_words)

# Count number of punctuations
punctuations="""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
def count_puncts(text):
    d = dict()

    for i in punctuations:
        d[str(i) + ' count'] = text.count(i)

    return d

# Count numbers in text
def count_numericals(text):
    numericals = map(str.isnumeric, text.split())
    return sum(numericals)
        

# Count number of words inside quotes
def count_quotedwords(text):
    x = re.findall("'.'|\".\"", text)

    if x is None:
        return 0
    else:
        count = 0
        for q_sent in x:
            sent = q_sent[1:-1] # Take the word after starting quote and before beginning quote
            count += count_words(sent)
        return count

# Count sentences in the text
def count_sents(text):
    sents = sent_tokenize(text)
    return len(sents)

# Count number of unque words
def count_uniquewords(text):
    unq_words = set(text.split())
    return len(unq_words)

# Count number of hashtags
def count_htags(text):
    x = re.findall(r'(#w[A-Za-z0-9]*)', text)

    return len(x)

# Count number of mentions
def count_mentions(text):
    x = re.findall(r'(@w[A-Za-z0-9]*)', text)

    return len(x)

# Count number of stopwords
def count_stopwords(text):
    word_tokens = word_tokenize(text)
    stopwords_x = [token for token in word_tokens if token not in stop_words]
    return len(stopwords_x)


# Preprocess the text using the strategy
def preprocess(text):
    text = re.sub(r'(@w[A-Za-z0-9]*)', '', text)
    text = re.sub(r'(#w[A-Za-z0-9]*)', '', text)
    
    text = re.sub(r'([0-9]+)', '', text)
    text = re.sub('['+punctuations+']', '', text)
    
    text = re.sub(r's+', ' ', text)
    
    text = text.lower()
    
    tokens = [stemmer.stem(token) for token in word_tokenize(text) if token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to prepare the news data for the model
def prepare_model_input(news_data):
    # Clean the news
    news_data['text'] = news_data['text'].apply(clean_news)
    
    # Generate the features
    news_data['num_characters'] = news_data['text'].apply(count_characters)
    news_data['num_words'] = news_data['text'].apply(count_words)
    news_data['num_capital_characters'] = news_data['text'].apply(count_capchars)
    news_data['num_capital_words'] = news_data['text'].apply(count_capwords)
    news_data['num_punctuations'] = news_data['text'].apply(count_puncts)
    news_data['num_numericals'] = news_data['text'].apply(count_numericals)
    news_data['num_quote_words'] = news_data['text'].apply(count_quotedwords)
    news_data['num_sents'] = news_data['text'].apply(count_sents)
    news_data['num_unique_words'] = news_data['text'].apply(count_uniquewords)
    news_data['num_htags'] = news_data['text'].apply(count_htags)
    news_data['num_mentions'] = news_data['text'].apply(count_mentions)
    news_data['num_stopwords'] = news_data['text'].apply(count_stopwords)

    # Calculate features from existing features
    news_data['avg_word_length'] = news_data['num_characters'] / news_data['num_words']
    news_data['avg_sent_length'] = news_data['num_words'] / news_data['num_sents']
    news_data['unique_vs_words'] = news_data['num_unique_words'] / news_data['num_words']
    news_data['stopwords_vs_words'] = news_data['num_stopwords'] / news_data['num_words']

    # Add punctuations as features
    puncts_data = pd.DataFrame(list(news_data['num_punctuations']))
    news_data = pd.merge(left=news_data, right=puncts_data, left_index=True, right_index=True)
    news_data = news_data.drop(columns=['num_punctuations'])

    # Apply preprocessing on the news
    news_data['text'] = news_data['text'].apply(preprocess)
    
    # Load the vectorizer
    vectorizer = joblib.load('../model/vectorizer.joblib')

    # Convert the news into vectors
    vectorized_features = vectorizer.transform(news_data['text'])

    # Take the relevant features
    text_features = ['num_characters', 'num_words',
       'num_capital_characters', 'num_capital_words', 'num_numericals',
       'num_quote_words', 'num_sents', 'num_unique_words', 'num_htags',
       'num_mentions', 'num_stopwords', 'avg_word_length', 'avg_sent_length',
       'unique_vs_words', 'stopwords_vs_words', '! count', '" count',
       '# count', '$ count', '% count', '& count', "' count", '( count',
       ') count', '* count', '+ count', ', count', '- count', '. count',
       '/ count', ': count', '; count', '< count', '= count', '> count',
       '? count', '@ count', '[ count', '\ count', '] count', '^ count',
       '_ count', '` count', '{ count', '| count', '} count', '~ count']

    # Merge the relevant features with the vectorized features
    news_data = pd.merge(left=news_data[text_features], right=pd.DataFrame(vectorized_features.toarray()), left_index=True, right_index=True)

    return news_data.to_numpy()

def make_predictions(X):
    pipeline = joblib.load('../model/pipeline.joblib')
    predictions = pipeline.predict(X)

    return predictions

def main():
    #Setting Application title
    st.title('Fake News Detector')

    # Setting application description
    st.markdown("""
    :dart: This streamlit application is built to do classification on news and decide whether a news is fake or real.
    """)
    st.markdown("This application provides both batch and online prediction.")

    # Build the sidebars of the streamlit app
    image = Image.open("../images/app-image.png")
    add_selectbox = st.sidebar.selectbox("How do you want to do the classification?", ("Online", "Batch"))
    st.sidebar.info('This application is created to detect fake news.')
    st.sidebar.image(image)

    if add_selectbox=="Online":
        st.subheader("Paste the news to check")
        news = st.text_area("")
        news = {'text': [news]}
        news = pd.DataFrame.from_dict(news)

        model_in = prepare_model_input(news)

        if st.button('Predict'):
            model_out = make_predictions(model_in)

            if model_out[0] == 1:
                st.warning("This news is fake!")
            else:
                st.success("This news is real.")

    elif add_selectbox=="Batch":
        st.info("Upload excel file containing the list of news")
        st.markdown("Make sure the excel file contains a table that has only one column named 'text'.")
        uploaded_file = st.file_uploader("choose an excel file")
        if uploaded_file is not None:
            news = pd.read_excel(uploaded_file)
            copy_news = news.copy()
            st.write(copy_news.head())

            model_in = prepare_model_input(copy_news)
            
            if st.button('Predict'):
                model_out = make_predictions(model_in)

                predictions = pd.DataFrame({'Text':news['text'], 'Prediction': model_out})

                st.subheader('Predictions')
                predictions['Prediction'] = predictions['Prediction'].replace({0: 'Real', 1: 'Fake'})
                st.write(predictions)

if __name__ == '__main__':
        main()
