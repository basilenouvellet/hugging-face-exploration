# Hugging Face Exploration

Just exploring NLP using 🤗 Transformers.

For now, only French text generation with CamemBERT is implemented.

## Install

```bash
source env/bin/activate
pip install -r requirements.txt
```

## Text generation

> For now, only available in **French** with CamemBERT, adapted from RoBERTa.

Just run one of:

```bash
python generate.py
```

Or 

```bash
python generate.py --text "Ton texte ici est <mask> et fun."
```

Or

```bash
python generate.py -t "Ton texte ici est <mask> et fun."
```

> If `<mask>` is at the end of the text, you can omit it:
>
> `python generate.py -t "Je m'appelle"`
