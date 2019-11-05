#!/usr/bin/env python
# coding: utf-8

import argparse
import wikipediaapi
import nltk
import pickle 
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow 

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to determinee whether a statement needs a citation or not.')
    parser.add_argument('-n', '--section_name', help='The input wikipedia article name on which predictions will be made.', required=True)
    parser.add_argument('-o', '--out_dir', help='The output directory where we store the results', required=False)
    parser.add_argument('-m', '--model', help='The path to the model which we use for classifying the statements.', required=True)
    parser.add_argument('-v', '--vocab', help='The path to the vocabulary of words we use to represent the statements.', required=True)
    parser.add_argument('-s', '--sections', help='The path to the vocabulary of section with which we trained our model.', required=True)
    parser.add_argument('-l', '--lang', help='The language that we are parsing now.', required=False, default='en')

    return parser.parse_args()

# code adapted from run_citation_need_model.py scrip
def construct_instance_reasons(statement, section_dict_path, vocab_w2v_path, max_len=-1):
    # Load the vocabulary
    vocab_w2v = pickle.load(open(vocab_w2v_path, 'rb'),encoding='latin1')

    # load the section dictionary.
    section_dict = pickle.load(open(section_dict_path, 'rb'))

    # construct the training data
    X = []
    sections = []
    y = []
    outstring=[]
    
    X_inst = []

    for word in statement:
        if max_len != -1 and len(X_inst) >= max_len:
            continue
        if word not in vocab_w2v:
            X_inst.append(vocab_w2v['UNK'])
        else:
            X_inst.append(vocab_w2v[word])

    # extract the section, and in case the section does not exist in the model, then assign UNK
    section = statement.strip().lower()
    sections.append(np.array([section_dict[section] if section in section_dict else 0]))

    X.append(X_inst)
    outstring.append(statement)
    #entity_id  revision_id timestamp   entity_title    section_id  section prg_idx sentence_idx    statement   citations

    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')

    return X, np.array(sections), outstring


def get_phrases_by_section(section, title):
    if len(section.sections) == 0:
        title.append(section.title)
        if section.text != '':
            return [(title,phrase) for phrase in sent_detector.tokenize(section.text.strip())]
        else:
            return []
    
    result = []
    title.append(section.title)
    for subsection in section.sections:
        t = title.copy()
        response = get_phrases_by_section(subsection, t)
        if len(response) > 0:
            result += response
    
    return result
    
def make_predictions(model, phrases_by_section, word_dict_path, section_dic_path):
    predictions = []
    max_seq_length = model.input[0].shape[1].value

    for section, phrase in phrases_by_section:
        X, sections, outstring = construct_instance_reasons(phrase, "embeddings/word_dict_en.pck", "embeddings/section_dict_en.pck", max_seq_length)
        pred = model.predict([X, sections])
        predictions.append([pred, phrase])
    
    predictions.sort(key=lambda tup: tup[0][0][0]) 
    return predictions


nltk.download('punkt') # at first run
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

def main():
    arguments = get_arguments()
    wiki_page = wiki_wiki.page(arguments.section_name)
    path_model = arguments.model
    model = load_model(path_model)
    phrases_by_section = get_phrases_by_section(wiki_page,[])
    predictions = make_predictions(model, phrases_by_section, arguments.vocab, arguments.sections)
    print("(prediction) PHRASE")
    for prediction in predictions:
        print("(" + str(prediction[0][0][0] + ") " + prediction[1]))


if __name__ == "__main__":
    main()




