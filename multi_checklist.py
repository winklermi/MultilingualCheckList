import argparse
from checklist.editor import Editor
from checklist.test_types import MFT
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt

def suggest_words():
    """get adj and noun suggestions from the CheckList Editors in German and English"""
    # initialise CheckList Editor objects
    eng_editor = Editor() # default language is English
    deu_editor = Editor(language="german")

    # get adj suggestions from the CheckList Editors in German and English
    print("English adj suggestions: ")
    eng_adj = eng_editor.suggest("This is a {mask} {noun}.", noun=["book", "movie"])[:20]
    print(eng_adj, "\n") 

    print("German adj suggestions: ")
    deu_adj = deu_editor.suggest("Das ist ein {mask} Buch.")[:20]
    deu_adj += deu_editor.suggest("Das ist ein {mask} Film.")[:20]
    print(deu_adj) 
    print(deu_adj, "\n")

    # get noun suggestions from the CheckList Editors in German and English
    # english suggestions
    print("English noun suggestions: ")
    print(eng_editor.suggest("This {mask} is {adj}.", adj=eng_adj)[:20], "\n") 

    # suggestions for standard German with different determiners
    print("German noun suggestions: ")
    print("Der", deu_editor.suggest("Der {mask} ist {adj}.", adj=deu_adj)[:20])
    print("Die", deu_editor.suggest("Die {mask} ist {adj}.", adj=deu_adj)[:20])
    print("Das", deu_editor.suggest("Das {mask} ist {adj}.", adj=deu_adj)[:20])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--suggest", action = "store_true")
    parser.add_argument("-r", "--run", action = "store_true")
    args = parser.parse_args()

    if args.suggest:
        # get adj and noun suggestions
        suggest_words()

    if args.run:
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


