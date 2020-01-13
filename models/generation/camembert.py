import torch
from transformers import CamembertTokenizer, CamembertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

def run_generation(text):
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
    print('\nTokenized text: %s (%s)' %(tokenized_text, tokenizer.decode(tokenized_text)))

    input_ids = torch.tensor(tokenized_text).unsqueeze(0)

    model = CamembertForMaskedLM.from_pretrained("camembert-base", resume_download=True)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids)

        masked_index = (input_ids.squeeze() == tokenizer.mask_token_id).nonzero().item()
        print('Masked index: %d (%s)\n' % (masked_index, tokenizer.mask_token))

        last_hidden_states = outputs[0]
        logits = last_hidden_states[0, masked_index, :]
        prob = logits.softmax(dim=0)
        values, indices = prob.topk(k=5, dim=0)

        return list(zip(
            [tokenizer.decode([x]) for x in indices],
            [round(v.item(), 2) for v in values]
        ))
