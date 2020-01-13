import os

# String templates for logging results
LOG_HEADER = 'Split  Dev/P_30  Dev/MAP  Dev/MRR  Dev/Loss'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:7.4f}'.split(','))
DEV_LOG_HEADER = 'Epoch Iteration Progress   Dev/P_30  Dev/MAP  Dev/MRR   Dev/Loss'
DEV_LOG_TEMPLATE = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:10.4f}'.split(','))

# Path to pretrained model files
MODEL_DATA_DIR = os.path.join(os.pardir, 'data', 'models')
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-uncased'),
    'bert-large-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-uncased'),
    'bert-base-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-cased'),
    'bert-large-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-cased'),
    'bert-base-multilingual-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-uncased'),
    'bert-base-multilingual-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-cased')
}

# Path to pretrained vocab files
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-uncased-vocab.txt'),
    'bert-large-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-uncased-vocab.txt'),
    'bert-base-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-cased-vocab.txt'),
    'bert-large-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-large-cased-vocab.txt'),
    'bert-base-multilingual-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-uncased-vocab.txt'),
    'bert-base-multilingual-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-cased-vocab.txt')
}

# Path to word2vec embedding files
WORD2VEC_EMBEDDING_FILE = 'GoogleNews-vectors-negative300.txt'
WORD2VEC_EMBEDDING_DIR = os.path.join(MODEL_DATA_DIR, 'word2vec')

# Path to TREC eval binary
TREC_EVAL_PATH = os.path.join(os.pardir, 'bin', 'trec_eval', 'trec_eval')

# Path to dataset directory
DATASET_DIR = os.path.join(os.pardir, 'data', 'datasets')

# Path to dataset qrel files
QREL_PATH_MAP = {
    'microblog': os.path.join(DATASET_DIR, 'microblog', 'trec_mb_qrels.txt')
}
