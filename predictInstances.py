from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import traduceAndRandomize
def predict(pathTest):
    MODEL_PATH = './models/'
    TOKENIZER = 'cardiffnlp/twitter-xlm-roberta-base'

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True, model_max_length=512) # Model max length is not set...
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
    #print(model)

    test = pd.read_csv(pathTest)
    texts = test['text'].to_list()
    tiny_texts = test['text'][:100].to_list()

    test_tokens = tokenizer(tiny_texts, truncation=True, padding=True, return_tensors='pt', max_length=512, add_special_tokens=True)


    with torch.no_grad(): # Disabling gradient calculation is useful for inference, when you are sure that you will not update weigths
        outputs = model(**test_tokens)[0]

    correct = 0
    for index, output in enumerate(outputs):
        if output[0] > output[1]: predicted = 'suicide'
        else: predicted = 'non-suicide'

        print(f'{test["text"][index][:30]}; real: {test["class"][index]}; probabilidad: {predicted}')

        if test["class"][index] == predicted: correct += 1

    print(correct)


if __name__ == '__main__':
    # LANGUAGE
    languages=['en','es','fr','it','ca','eu','bg','hy','ka','ug']
    for language in languages:
        traduceAndRandomize.traduce('formated/test.csv',f'traduced/test_{language}.csv',language)
        predict(f'traduced/test_{language}.csv')

    # RANDOMIZED
    predict('formated/test.csv')
    traduceAndRandomize.randomize('formated/test.csv','randomized_test.csv')
    predict('randomized_test.csv')