import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1

sys.path.append(os.getcwd())

import language_helpers
from gan_language import (
    DEFAULT_DATA_ROOT,
    build_generator,
    decode_samples,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DNA GAN checkpoints on sequence_craft JSD metrics')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint prefix to restore')
    parser.add_argument('--label', required=True, help='Human-readable label for this checkpoint')
    parser.add_argument('--dataset', default='sequence_craft', help='Reference dataset for n-gram models')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--sample-rounds', type=int, default=5,
                        help='Number of sample batches (each round = 10 forward passes, like training)')
    parser.add_argument('--output', default=None, help='Append JSON results to this file')
    return parser.parse_args()


def build_true_ngram_models(lines, batch_size):
    ngram_sample_size = min(len(lines), 20000)
    if len(lines) <= ngram_sample_size:
        ngram_lines = lines
    else:
        ngram_lines = [lines[i] for i in np.random.choice(len(lines), ngram_sample_size, replace=False)]

    return [
        language_helpers.NgramLanguageModel(i + 1, ngram_lines, tokenize=False)
        for i in range(4)
    ]


def evaluate_checkpoint(args):
    tfv1.disable_eager_execution()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    lines, charmap, inv_charmap = language_helpers.load_dataset(
        max_length=args.seq_len,
        max_n_examples=None,
        dataset=args.dataset,
        data_root=args.data_root,
    )
    vocab_size = len(charmap)
    true_char_ngram_lms = build_true_ngram_models(lines, args.batch_size)

    fake_inputs = build_generator(args.batch_size, args.seq_len, args.dim, vocab_size)

    session_config = tfv1.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.graph_options.optimizer_options.global_jit_level = tfv1.OptimizerOptions.OFF

    round_metrics = []
    start = time.time()

    with tfv1.Session(config=session_config) as session:
        saver = tfv1.train.Saver()
        saver.restore(session, args.checkpoint)
        print(f"Restored checkpoint: {args.checkpoint}")

        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            return decode_samples(samples, inv_charmap)

        for round_idx in range(args.sample_rounds):
            np.random.seed(round_idx)
            samples = []
            for _ in range(10):
                samples.extend(generate_samples())

            metrics = {}
            for i in range(4):
                lm = language_helpers.NgramLanguageModel(i + 1, samples, tokenize=False)
                key = f'js{i + 1}'
                metrics[key] = float(lm.js_with(true_char_ngram_lms[i]))
                metrics[f'precision{i + 1}'] = float(lm.precision_wrt(true_char_ngram_lms[i]))
                metrics[f'recall{i + 1}'] = float(true_char_ngram_lms[i].precision_wrt(lm))

            metrics['js_sum'] = sum(metrics[f'js{i + 1}'] for i in range(4))
            metrics['n_samples'] = len(samples)
            round_metrics.append(metrics)
            print(
                f"  round {round_idx + 1}/{args.sample_rounds}: "
                f"js1={metrics['js1']:.6f} js2={metrics['js2']:.6f} "
                f"js3={metrics['js3']:.6f} js4={metrics['js4']:.6f} "
                f"sum={metrics['js_sum']:.6f}"
            )

    summary = {
        'label': args.label,
        'checkpoint': args.checkpoint,
        'dataset': args.dataset,
        'sample_rounds': args.sample_rounds,
        'elapsed_sec': time.time() - start,
    }

    for key in round_metrics[0]:
        values = [m[key] for m in round_metrics]
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values))

    summary['rounds'] = round_metrics
    return summary


def main():
    args = parse_args()
    print(f"=== Evaluating: {args.label} ===")
    result = evaluate_checkpoint(args)

    print(
        f"MEAN js1={result['js1_mean']:.6f} js2={result['js2_mean']:.6f} "
        f"js3={result['js3_mean']:.6f} js4={result['js4_mean']:.6f} "
        f"sum={result['js_sum_mean']:.6f} "
        f"(±{result['js4_std']:.6f} on js4)"
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        if os.path.exists(args.output):
            with open(args.output) as f:
                all_results = json.load(f)
        else:
            all_results = []
        all_results.append(result)
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
