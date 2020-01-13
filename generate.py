import pprint
import argparse
from models.generation.camembert import run_generation

parser = argparse.ArgumentParser(description='Run CamemBERT on a sentence with a `<mask>` word.')
parser.add_argument('-t', '--text', dest='text', type=str,
                    help='the text containing a `<mask>` word')

def sanitize_text(text):
    # auto-append <mask> at the end if not present
    if '<mask>' not in text:
        text = text + ' <mask>.'
    # mask is at the end and no punctuation is there
    if text[-6:] == '<mask>':
        text = text + '.'
    return text

def get_text_from_user():
    DEFAULT_TEXT = 'Le camembert est  <mask> !'
    print((
        'Entrer une phrase avec `<mask>` à la place du mot à remplacer\n'
        'Optionnel si `<mask>` se situe à la fin\n'
    ))
    text = input()
    return text if text else DEFAULT_TEXT

def get_text():
    args = parser.parse_args()
    
    if args.text is None:
        text = get_text_from_user()
    else:
        text = args.text
    
    return sanitize_text(text)

def display_results(results, text):
    res = [(text.replace('<mask>', word), proba) for (word, proba) in results]
    pprint.pprint(res)

if __name__ == "__main__":
    text = get_text()
    results = run_generation(text)

    display_results(results, text)
