import shlex
import subprocess
from collections import defaultdict

from tqdm import tqdm

from common.constants import TREC_EVAL_PATH


def save_ranks(output_path, query_ids, doc_ids, scores):
    # Aggregate scores by query_id
    scores_by_query = defaultdict(list)
    for query_id, doc_id, score in zip(query_ids, doc_ids, scores):
        scores_by_query[query_id].append((doc_id, score))

    with open(output_path, 'w') as output_file:
        for query_id in tqdm(scores_by_query, desc='Saving'):
            # Aggregate scores by doc_id
            max_scores = defaultdict(list)
            for doc_id, score in scores_by_query[query_id]:
                max_scores[doc_id].append(score)

            sorted_score = sorted(((sum(scores) / len(scores), doc_id)
                                   for doc_id, scores in max_scores.items()),
                                  reverse=True)

            rank = 1  # Reset rank counter to one
            for score, doc_ids in sorted_score:
                output_file.write(f'{query_id} Q0 {doc_ids} {rank} {score} Errol\n')
                rank += 1


def run_trec_eval(qrels_path, ranks_path):
    cmd = '%s %s %s -m map -m P.30 -m recip_rank' % (TREC_EVAL_PATH, qrels_path, ranks_path)
    pargs = shlex.split(cmd)
    p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pout, perr = p.communicate()
    pout = [line.split() for line in pout.split(b'\n') if line.strip()]

    metrics = dict()
    for metric, _, value in pout:
        metrics[metric.decode("utf-8").lower()] = float(value)

    return metrics
