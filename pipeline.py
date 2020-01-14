from transformers import pipeline

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

nlp = pipeline('question-answering')

qa_result = nlp({
    'question': 'What is the name of the repository ?',
    'context': 'Pipeline have been included in the huggingface/transformers repository'
})
print('QA:', qa_result)

qa_result = nlp({
    'question': 'What do my mum like to eat ?',
    'context': 'I have 2 sisters and 4 brothers. My mum prefers apples than pears. My mum like her dress'
})
print('QA:', qa_result)
