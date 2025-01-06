from checklist.editor import Editor
from checklist.test_types import MFT
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt

# use vanilla mBERT for the same analysis process for comparison

# initialise CheckList Editor objects
eng_editor = Editor() # default language is English
deu_editor = Editor(language="german")

# parallel adjective lists for data creation
eng_pos = ["good", "nice", "great"]
deu_pos = ["gut", "schön", "super"]
bar_pos = ["guad", "schee", "subba"]

eng_neg = ["bad", "boring", "stupid"]
deu_neg = ["schlecht", "langweilig", "blöd"]
bar_neg = ["schlecht", "fad", "bled"]

# parallel noun lists for data creation
# picked from Editor suggestions in such a way that mostly all words are different
eng_noun = ["game", "site", "picture", "book", "story", "man", "world", "city", "time", "weather", "life"]

# standard German and Bavarian examples with determiners to avoid errors
deu_noun = [
    ("Das", "Spiel"), ("Die", "Seite"), ("Das", "Bild"), ("Das", "Buch"), ("Die", "Geschichte"), 
    ("Der", "Mann"), ("Die", "Welt"), ("Die", "Stadt"), ("Die", "Zeit"), ("Das", "Wetter"), ("Das", "Leben")
    ]

# bavarian determiners are with spaces to handle "d'" and "s'" determiners
bar_noun = [
    ("Des ", "Spui"), ("De ", "Seitn"), ("Des ", "Buidl"), ("Des ", "Buach"), ("De ", "Gschicht"), 
    ("Der ", "Mo"), ("D'", "Weid"), ("D'", "Stod"), ("D'", "Zeid"), ("S'", "Weda"), ("S'", "Lebm")
    ] 

# create english data
v_eng = eng_editor.template("The {noun} is not {adj}.", noun=eng_noun, adj=eng_pos, labels=0) # not pos = negative
v_eng += eng_editor.template("The {noun} is not {adj}.", noun=eng_noun, adj=eng_neg, labels=1) # not neg = positive

# create german data
v_deu = deu_editor.template("{detnoun[0]} {detnoun[1]} ist nicht {adj}.", detnoun=deu_noun, adj=deu_pos, labels=0)
v_deu += deu_editor.template("{detnoun[0]} {detnoun[1]} ist nicht {adj}.", detnoun=deu_noun, adj=deu_neg, labels=1)

# create bavarian data
v_bar = deu_editor.template("{detnoun[0]}{detnoun[1]} is ned {adj}.", detnoun=bar_noun, adj=bar_pos, labels=0)
v_bar += deu_editor.template("{detnoun[0]}{detnoun[1]} is ned {adj}.", detnoun=bar_noun, adj=bar_neg, labels=1)


# initialise minimum functionality tests
v_eng_test = MFT(**v_eng, name="English Negations", capability="Negation", description="Litotes sentences to test negation capabilities for vanilla mBERT.")
v_deu_test = MFT(**v_deu, name="Standard German Negations", capability="Negation", description="Litotes sentences to test negation capabilities for vanilla mBERT.")
v_bar_test = MFT(**v_bar, name="Bavarian Negations", capability="Negation", description="Litotes sentences to test negation capabilities for vanilla mBERT.")


# load vanilla mBERT model
v_model_name = "google-bert/bert-base-multilingual-cased"
v_tokenizer = AutoTokenizer.from_pretrained(v_model_name)
v_model = AutoModelForSequenceClassification.from_pretrained(v_model_name)

# initialise pipeline for predictions
device = "cuda" if torch.cuda.is_available() else "cpu"
v_pipe = pipeline("sentiment-analysis", model=v_model, tokenizer=v_tokenizer, device=device, return_all_scores=True)

# make predictions on test data
v_lbl2idx = {"LABEL_0": 0, "LABEL_1": 1}
v_idx2lbl = {0: "LABEL_0", 1: "LABEL_1"}

def v_predict(test):
    # read data and predict
    data = test.to_raw_examples() # necessary for internal CheckList structures (self.result_indexes)
    raw_preds = v_pipe(data)

    preds = []
    confs = []

    # write results in correct CheckList format to a file
    for result in raw_preds:
        negative = result[0]
        positive = result[1]

        max_pred = max([negative, positive], key=lambda x: x["score"])
        max_label = max_pred["label"]

        # prediction, negative_score, neutral_score, positive_score
        preds.append(v_lbl2idx[max_label])
        confs.append(np.array([negative["score"], positive["score"]]))

    return preds, confs

v_eng_preds, v_eng_confs = v_predict(v_eng_test)
v_deu_preds, v_deu_confs = v_predict(v_deu_test)
v_bar_preds, v_bar_confs = v_predict(v_bar_test)


# visualise vanilla mBERT results
def v_visualise(golds, preds):
    # from integer labels to text labels
    gold_labels = [v_idx2lbl[label] for label in golds]
    pred_labels = [v_idx2lbl[label] for label in preds]
    
    classes = v_lbl2idx.keys()
    num_classes = np.arange(len(v_lbl2idx)) # spaced out for pyplot

    # extract counts and calculate percentages of correct and false predictions
    correct_preds_per_label = {label:0 for label in classes}
    false_preds_per_label = {label:0 for label in classes}

    for label in classes:
        for gold, pred in zip(gold_labels, pred_labels):
            if gold == label and gold == pred: # if gold is the current label and a correct prediction
                correct_preds_per_label[label] += 1

            else: 
                false_preds_per_label[label] += 1

    correct_percentages = [(count / len(gold_labels)) * 100 for count in correct_preds_per_label.values()]
    false_percentages = [(count / len(gold_labels)) * 100 for count in false_preds_per_label.values()]
    
    # design bar chart
    width = 0.8
    plt.bar(num_classes, correct_percentages, width, label = "Correct", color = "mediumseagreen")
    plt.bar(num_classes, false_percentages, width, bottom=correct_percentages, label = "False", color = "salmon")

    plt.xlabel("Labels")
    plt.ylabel("Percentage")
    plt.title("Correct and False Predictions per Label")
    plt.xticks(num_classes, classes)
    plt.legend()

    # plt.tight_layout()
    plt.show()


# run eng tests
v_eng_test.run_from_preds_confs(v_eng_preds, v_eng_confs, overwrite=True)
v_eng_test.summary()

v_visualise(v_eng.labels, v_eng_preds)

# run deu tests
v_deu_test.run_from_preds_confs(v_deu_preds, v_deu_confs, overwrite=True)
v_deu_test.summary()

v_visualise(v_deu.labels, v_deu_preds)

# run bar tests
v_bar_test.run_from_preds_confs(v_bar_preds, v_bar_confs, overwrite=True)
v_bar_test.summary()

v_visualise(v_bar.labels, v_bar_preds)