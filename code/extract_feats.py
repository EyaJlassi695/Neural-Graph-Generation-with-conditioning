# extract_feats.py
import random
import re

random.seed(32)

def extract_numbers(text, model, tokenizer):
    output = model(
        **tokenizer.batch_encode_plus(
            [
                (
                    text
                )
            ],
        padding='longest',
        add_special_tokens=True,
        return_tensors='pt'
        )
    )
    embeddings = output.pooler_output
    return embeddings


def extract_numbers_simple(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]
def extract_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    fread.close()
    return line