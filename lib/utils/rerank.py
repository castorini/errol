import os
from collections import defaultdict

import numpy as np

from common.constants import RERANK_LOG_HEADER, RERANK_LOG_TEMPLATE
from utils.io import run_trec_eval


def evaluate_split(qrels_path, ranks_path, alpha, split):
    metrics = run_trec_eval(qrels_path, ranks_path)
    print('\n' + RERANK_LOG_HEADER)
    print(RERANK_LOG_TEMPLATE.format(split.upper(), alpha, metrics['p_30'], metrics['map'], metrics['recip_rank']))
    return metrics


def load_ranks(ranks_path):
    topics = set()
    score_dict = defaultdict(dict)

    with open(ranks_path, 'r') as f:
        for line in f:
            topic, _, docid, _, score, _ = line.split()
            topics.add(topic)
            score_dict[topic.strip()][docid.strip()] = float(score)

    return score_dict, topics


def merge_ranks(old_ranks, new_ranks, topics):
    doc_ranks = dict()
    print()  # Print blank line to separate output

    for topic in topics:
        missing_docids = list()
        old_scores = old_ranks[topic]
        new_scores = new_ranks[topic]

        if topic not in doc_ranks:
            doc_ranks[topic] = list(), list(), list()

        for docid, old_score in old_scores.items():
            try:
                new_score = new_scores[docid]
                doc_ranks[topic][0].append(docid)
                doc_ranks[topic][1].append(old_score)
                doc_ranks[topic][2].append(new_score)
            except KeyError:
                missing_docids.append(docid)

        print("Number of missing documents in topic %s: %d" % (topic, len(missing_docids)))

    return doc_ranks


def interpolate(old_scores, new_scores, alpha):
    s_min, s_max = min(old_scores), max(old_scores)
    old_score = (old_scores - s_min) / (s_max - s_min)
    s_min, s_max = min(new_scores), max(new_scores)
    new_score = (new_scores - s_min) / (s_max - s_min)
    score = old_score * (1 - alpha) + new_score * alpha
    return score


def rerank_alpha(doc_ranks, alpha, limit, filename, tag='Errol'):
    with open(os.path.join(filename), 'w') as f:
        for topic in doc_ranks:
            docids, old_scores, new_scores = doc_ranks[topic]
            score = interpolate(np.array(old_scores), np.array(new_scores), alpha)
            sorted_score = sorted(list(zip(docids, score)), key=lambda x: -x[1])

            rank = 1
            for docids, score in sorted_score:
                f.write(f'{topic} Q0 {docids} {rank} {score} {tag}\n')
                rank += 1
                if rank > limit:
                    break


def rerank(qrels_path, retrieval_ranks_path, dev_ranks_path, test_ranks_path):
    retrieval_ranks, retrieval_topics = load_ranks(retrieval_ranks_path)
    dev_ranks, dev_topics = load_ranks(dev_ranks_path)
    interpolated_ranks = merge_ranks(retrieval_ranks, dev_ranks, topics=dev_topics)

    best_alpha = 0
    best_dev_map = 0

    for alpha in range(10, 100, 10):
        alpha_f = alpha / 100  # Convert alpha from percentage to decimal
        interpolated_ranks_path = '%s_rerank_%0.1f.txt' % (os.path.splitext(dev_ranks_path)[0], alpha_f)
        rerank_alpha(interpolated_ranks, alpha_f, 10000, interpolated_ranks_path)
        metrics = evaluate_split(qrels_path, interpolated_ranks_path, alpha_f, split='dev')

        # Calculate the best interpolation factor on dev split
        if metrics['map'] > best_dev_map:
            best_alpha = alpha_f
            best_dev_map = metrics['map']

    # Rerank the test split with best_alpha as the interpolation factor
    test_ranks, test_topics = load_ranks(test_ranks_path)
    interpolated_ranks = merge_ranks(retrieval_ranks, test_ranks, topics=test_topics)
    interpolated_ranks_path = '%s_rerank_%0.1f.txt' % (os.path.splitext(test_ranks_path)[0], best_alpha)
    rerank_alpha(interpolated_ranks, best_alpha, 10000, interpolated_ranks_path)
    evaluate_split(qrels_path, interpolated_ranks_path, best_alpha, split='test')
