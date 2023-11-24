import pandas as pd
from gensim.summarization.summarizer import summarize
from gensim.utils import tokenize


def sumarize(text, word_count):
    
    summarized = summarize(text, word_count=word_count, split=True)
    summarized = ' '.join(summarized)
    return summarized



if __name__ == '__main__':

    data = pd.read_csv('./formated/Suicide_Detection10000.csv')
    textos = data['text']
    
    print(textos[24])
    print(len(textos[24]))