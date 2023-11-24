from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import traduceAndRandomize
from sklearn.metrics import classification_report

def predict(pathTest, language, numinstances):

    MODEL_PATH = '../models/'
    TOKENIZER = 'cardiffnlp/twitter-xlm-roberta-base'

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True, model_max_length=512) 
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)

    test = pd.read_csv(pathTest)
    labels = test['class'][:10].to_list()
    tiny_texts = test['text'][:10].to_list()

    test_tokens = tokenizer(tiny_texts, truncation=True, padding=True, return_tensors='pt', max_length=512, add_special_tokens=True)


    with torch.no_grad(): 
        outputs = model(**test_tokens)[0]

    predictions = []
    
    correct = 0
    for index, output in enumerate(outputs):
        if output[0] > output[1]: predicted = 'suicide'
        else: predicted = 'non-suicide'

        predictions = predictions + [predicted]

        # print(f'{test["text"][index][:30]}; real: {test["class"][index]}; probabilidad: {predicted}')

        if test["class"][index] == predicted: correct += 1

    with open(f'../results/report_{language}.txt', 'w') as file:

        file.write('################################\n')
        file.write(f'############### {language} ############# \n')
        file.write('################################\n\n')

        file.write('------ REPORT ------\n')
        
        print(classification_report(labels, predictions), file=file)

        file.write('\n------ CORRECTLY CLASSIFIED INSTANCES ------\n')

        print(str(correct) + ' / ' + str(numinstances), file=file)
