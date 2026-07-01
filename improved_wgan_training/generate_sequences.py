import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1
from tqdm import tqdm

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "FT"))

from gan_language import build_generator, decode_samples

try:
    import nupack  # noqa: F401
except ImportError:
    nupack = None

from mfe_utils import calculate_mfe


def parse_args():
    parser = argparse.ArgumentParser(description="Generate filtered DNA sequences from WGAN checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint prefix to restore")
    parser.add_argument("--label", required=True, help="Run label for output files")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--num-sequences", type=int, default=250000)
    parser.add_argument("--max-mfe", type=float, default=-10.0,
                        help="Keep sequences with MFE <= this value")
    parser.add_argument("--no-mfe-filter", action="store_true",
                        help="Do not filter by MFE; keep sequences that pass min-length only")
    parser.add_argument("--skip-mfe-calc", action="store_true",
                        help="Skip NUPACK MFE calculation (fastest; MFE column left empty)")
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=None,
                        help="Keep sequences with length < this value (e.g. 100 => max 99 bp)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-round", type=int, default=32,
                        help="GPU batches generated before MFE filtering")
    parser.add_argument("--mfe-workers", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=10000)
    return parser.parse_args()


def sample_to_string(sample):
    return "".join(sample).replace("`", "")


def passes_length(seq_str, min_length, max_length):
    n = len(seq_str)
    if n < min_length:
        return False
    if max_length is not None and n >= max_length:
        return False
    return True


def passes_filter(seq_str, max_mfe, min_length, max_length=None):
    if not passes_length(seq_str, min_length, max_length):
        return False, 0.0
    mfe = calculate_mfe(seq_str)
    return mfe <= max_mfe, mfe


def _length_only(seq_str, min_length, max_length, skip_mfe_calc):
    if not passes_length(seq_str, min_length, max_length):
        return None
    if skip_mfe_calc:
        return (seq_str, len(seq_str), "")
    mfe = calculate_mfe(seq_str)
    return (seq_str, len(seq_str), mfe)


def filter_sequences(raw_sequences, max_mfe, min_length, workers, max_length=None,
                     no_mfe_filter=False, skip_mfe_calc=False):
    accepted = []
    if no_mfe_filter:
        if skip_mfe_calc or workers <= 1:
            for seq_str in raw_sequences:
                row = _length_only(seq_str, min_length, max_length, skip_mfe_calc)
                if row:
                    accepted.append(row)
            return accepted

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_length_only, seq, min_length, max_length, skip_mfe_calc): seq
                for seq in raw_sequences
            }
            for future in as_completed(futures):
                row = future.result()
                if row:
                    accepted.append(row)
        return accepted

    if workers <= 1:
        for seq_str in raw_sequences:
            ok, mfe = passes_filter(seq_str, max_mfe, min_length, max_length)
            if ok:
                accepted.append((seq_str, len(seq_str), mfe))
        return accepted

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(passes_filter, seq, max_mfe, min_length, max_length): seq
            for seq in raw_sequences
        }
        for future in as_completed(futures):
            ok, mfe = future.result()
            if ok:
                seq_str = futures[future]
                accepted.append((seq_str, len(seq_str), mfe))
    return accepted


def write_rows(path, rows, write_header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["Sequence", "Length", "MFE"])
        writer.writerows(rows)


def generate(args):
    tfv1.disable_eager_execution()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    output_dir = args.output_dir or os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        "generated",
        args.label,
    )
    csv_path = os.path.join(output_dir, "generated_sequences.csv")
    stats_path = os.path.join(output_dir, "generation_stats.txt")

    fake_inputs = build_generator(args.batch_size, 100, 512, 6)
    session_config = tfv1.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.graph_options.optimizer_options.global_jit_level = tfv1.OptimizerOptions.OFF

    inv_charmap = ["unk", "`", "A", "T", "C", "G"]
    accepted_total = 0
    generated_total = 0
    pending_rows = []
    header_written = os.path.exists(csv_path)
    start = time.time()

    if args.no_mfe_filter and not args.skip_mfe_calc and nupack is None:
        raise SystemExit(
            "NUPACK is required when computing MFE without filtering. "
            "Use --skip-mfe-calc or install nupack."
        )
    if not args.no_mfe_filter and nupack is None:
        raise SystemExit(
            "NUPACK is required for MFE filtering. Install from https://www.nupack.org/download/software "
            "and run: pip install -U nupack -f /path/to/nupack-VERSION/package"
        )

    print(f"=== Generate {args.num_sequences} sequences ===")
    print(f"Checkpoint: {args.checkpoint}")
    if args.no_mfe_filter:
        mfe_note = "MFE not computed" if args.skip_mfe_calc else "MFE recorded, not filtered"
        length_note = f"{args.min_length} <= length"
        if args.max_length is not None:
            length_note += f" < {args.max_length}"
        print(f"Filter: {length_note} ({mfe_note})")
    else:
        length_note = f"length >= {args.min_length}"
        if args.max_length is not None:
            length_note += f", length < {args.max_length}"
        print(f"Filter: MFE <= {args.max_mfe}, {length_note}")
    print(f"Output: {csv_path}")

    with tfv1.Session(config=session_config) as session:
        saver = tfv1.train.Saver()
        saver.restore(session, args.checkpoint)
        print(f"Restored checkpoint: {args.checkpoint}")

        with tqdm(total=args.num_sequences, unit="seq") as progress:
            while accepted_total < args.num_sequences:
                raw_sequences = []
                for _ in range(args.batches_per_round):
                    samples = session.run(fake_inputs)
                    samples = np.argmax(samples, axis=2)
                    decoded = decode_samples(samples, inv_charmap)
                    raw_sequences.extend(sample_to_string(s) for s in decoded)

                generated_total += len(raw_sequences)
                accepted = filter_sequences(
                    raw_sequences,
                    args.max_mfe,
                    args.min_length,
                    args.mfe_workers,
                    max_length=args.max_length,
                    no_mfe_filter=args.no_mfe_filter,
                    skip_mfe_calc=args.skip_mfe_calc,
                )

                if accepted:
                    remaining = args.num_sequences - accepted_total
                    accepted = accepted[:remaining]
                    pending_rows.extend(accepted)
                    accepted_total += len(accepted)
                    progress.update(len(accepted))

                if len(pending_rows) >= args.flush_every or accepted_total >= args.num_sequences:
                    write_rows(csv_path, pending_rows, write_header=not header_written)
                    header_written = True
                    pending_rows = []

                rate_label = "keep_rate" if args.no_mfe_filter else "accept_rate"
                progress.set_postfix(
                    generated=generated_total,
                    **{rate_label: f"{100.0 * accepted_total / max(generated_total, 1):.2f}%"},
                )

    elapsed = time.time() - start
    lengths = []
    mfes = []
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lengths.append(int(row["Length"]))
            if row["MFE"] != "":
                mfes.append(float(row["MFE"]))

    with open(stats_path, "w") as handle:
        handle.write(f"label: {args.label}\n")
        handle.write(f"checkpoint: {args.checkpoint}\n")
        handle.write(f"num_sequences: {len(lengths)}\n")
        handle.write(f"generated_total: {generated_total}\n")
        handle.write(f"keep_rate: {accepted_total / max(generated_total, 1):.6f}\n")
        handle.write(f"no_mfe_filter: {args.no_mfe_filter}\n")
        handle.write(f"skip_mfe_calc: {args.skip_mfe_calc}\n")
        handle.write(f"max_mfe: {args.max_mfe}\n")
        handle.write(f"min_length: {args.min_length}\n")
        handle.write(f"max_length: {args.max_length}\n")
        handle.write(f"elapsed_sec: {elapsed:.1f}\n")
        handle.write(f"mean_length: {np.mean(lengths):.4f}\n")
        handle.write(f"std_length: {np.std(lengths):.4f}\n")
        if mfes:
            handle.write(f"mean_mfe: {np.mean(mfes):.4f}\n")
            handle.write(f"min_mfe: {np.min(mfes):.4f}\n")
            handle.write(f"max_mfe_value: {np.max(mfes):.4f}\n")

    print(f"Done: {len(lengths)} sequences -> {csv_path}")
    rate_name = "Keep rate" if args.no_mfe_filter else "Acceptance rate"
    print(f"{rate_name}: {100.0 * accepted_total / max(generated_total, 1):.2f}%")
    print(f"Elapsed: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    generate(parse_args())
