import re


def text_tokenize(string, max_length=5000):
    string = re.sub(r'[^A-Za-z0-9]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    tokenized_string = string.lower().strip().split()
    return tokenized_string[:min(max_length, len(tokenized_string))]


def binary_one_hot(label):
    if int(label) == 1:
        return [1.0, 0.0]
    else:
        return [0.0, 1.0]