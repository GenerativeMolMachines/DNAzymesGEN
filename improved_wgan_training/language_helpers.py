import collections
import os

import numpy as np
import pandas as pd

# Фиксированный словарь — одинаковый при pretrain и finetune для совместимости весов
DNA_CHARMAP = {'unk': 0, '`': 1, 'A': 2, 'T': 3, 'C': 4, 'G': 5}
DNA_INV_CHARMAP = ['unk', '`', 'A', 'T', 'C', 'G']

DATASET_PATHS = {
    'eds': {
        'type': 'csv',
        'path': 'EDS/distrib_result.csv',
        'column': 'sequence',
    },
    'mfe': {
        'type': 'csv',
        'path': 'MFE/seq2 - seq2.csv',
        'column': 'Sequence',
    },
    'sequence_craft': {
        'type': 'csv',
        'path': 'Sequence_Craft/SequenceCraft_dataset.csv',
        'column': 'e',
    },
}


def tokenize_string(sample):
    return tuple(sample.lower().split(' '))


class NgramLanguageModel(object):
    def __init__(self, n, samples, tokenize=False):
        if tokenize:
            tokenized_samples = []
            for sample in samples:
                tokenized_samples.append(tokenize_string(sample))
            samples = tokenized_samples

        self._n = n
        self._samples = samples
        self._ngram_counts = collections.defaultdict(int)
        self._total_ngrams = 0
        for ngram in self.ngrams():
            self._ngram_counts[ngram] += 1
            self._total_ngrams += 1

    def ngrams(self):
        n = self._n
        for sample in self._samples:
            for i in range(len(sample) - n + 1):
                yield sample[i:i + n]

    def unique_ngrams(self):
        return set(self._ngram_counts.keys())

    def log_likelihood(self, ngram):
        if ngram not in self._ngram_counts:
            return -np.inf
        return np.log(self._ngram_counts[ngram]) - np.log(self._total_ngrams)

    def kl_to(self, p):
        log_likelihood_ratios = []
        for ngram in p.ngrams():
            log_likelihood_ratios.append(p.log_likelihood(ngram) - self.log_likelihood(ngram))
        return np.mean(log_likelihood_ratios)

    def cosine_sim_with(self, p):
        p_dot_q = 0.
        p_norm = 0.
        q_norm = 0.
        for ngram in p.unique_ngrams():
            p_i = np.exp(p.log_likelihood(ngram))
            q_i = np.exp(self.log_likelihood(ngram))
            p_dot_q += p_i * q_i
            p_norm += p_i ** 2
        for ngram in self.unique_ngrams():
            q_i = np.exp(self.log_likelihood(ngram))
            q_norm += q_i ** 2
        return p_dot_q / (np.sqrt(p_norm) * np.sqrt(q_norm))

    def precision_wrt(self, p):
        num = 0.
        denom = 0
        p_ngrams = p.unique_ngrams()
        for ngram in self.unique_ngrams():
            if ngram in p_ngrams:
                num += self._ngram_counts[ngram]
            denom += self._ngram_counts[ngram]
        return float(num) / denom

    def recall_wrt(self, p):
        return p.precision_wrt(self)

    def js_with(self, p):
        log_p = np.array([p.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in p.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_p_m = np.sum(np.exp(log_p) * (log_p - log_m))

        log_p = np.array([p.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_q = np.array([self.log_likelihood(ngram) for ngram in self.unique_ngrams()])
        log_m = np.logaddexp(log_p - np.log(2), log_q - np.log(2))
        kl_q_m = np.sum(np.exp(log_q) * (log_q - log_m))

        return 0.5 * (kl_p_m + kl_q_m) / np.log(2)


def _read_raw_sequences(dataset, data_root, max_n_examples):
    if dataset not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_PATHS)}")

    spec = DATASET_PATHS[dataset]
    file_path = os.path.join(data_root, spec['path'])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if spec['type'] == 'csv':
        df = pd.read_csv(file_path)
        raw_lines = df[spec['column']].astype(str).tolist()
    elif spec['type'] == 'txt':
        with open(file_path, 'r') as f:
            raw_lines = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported dataset type: {spec['type']}")

    return raw_lines if max_n_examples is None or max_n_examples <= 0 else raw_lines[:max_n_examples]


def _encode_sequence(seq, max_length, tokenize=False):
    seq = seq.upper().strip()
    if tokenize:
        line = tokenize_string(seq)
    else:
        line = tuple(seq)

    if len(line) > max_length:
        line = line[:max_length]

    padded_line = line + ('`',) * (max_length - len(line))
    return padded_line


def load_dataset(max_length, max_n_examples, dataset='sequence_craft',
                 data_root='/root/dnazymes/Data', tokenize=False, max_vocab_size=2048):
    print(f"loading dataset '{dataset}' from {data_root}...")

    raw_lines = _read_raw_sequences(dataset, data_root, max_n_examples)
    lines = [_encode_sequence(line, max_length, tokenize=tokenize) for line in raw_lines]
    np.random.shuffle(lines)

    charmap = dict(DNA_CHARMAP)
    inv_charmap = list(DNA_INV_CHARMAP)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(min(5, len(filtered_lines))):
        print(filtered_lines[i])

    print(f"loaded {len(filtered_lines)} lines in dataset '{dataset}'")
    return filtered_lines, charmap, inv_charmap
