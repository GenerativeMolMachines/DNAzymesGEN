#!/usr/bin/env python3
"""
DNA Metrics Calculation with Multiprocessing
Optimized for large datasets - NO FULL PAIR LIST IN MEMORY
"""

import numpy as np
from itertools import islice
from collections import Counter
from multiprocessing import Pool, cpu_count
import random
import pandas as pd


# ==================== GLOBAL SETTINGS ====================
K_MER_SIZE = 3


# ==================== WORKER FUNCTIONS (TOP-LEVEL) ====================

def jaccard_distance(seq1, seq2, k=K_MER_SIZE):
    """Calculate Jaccard distance for k-mers"""
    kmers1 = set([seq1[i:i+k] for i in range(len(seq1)-k+1)])
    kmers2 = set([seq2[i:i+k] for i in range(len(seq2)-k+1)])
    intersection = len(kmers1 & kmers2)
    union = len(kmers1 | kmers2)
    return 1.0 - (intersection / union) if union > 0 else 1.0


def process_jaccard_chunk(args):
    """Worker function for multiprocessing - must be top-level"""
    chunk, k = args
    return [jaccard_distance(s1, s2, k) for s1, s2 in chunk]


# ==================== MEMORY-EFFICIENT METRICS ====================

def calculate_apjd_sampled(generated_seqs, train_seqs, k=K_MER_SIZE, n_samples=100000, n_processes=None):
    """
    APJD: Sample random pairs instead of all pairs
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    gen_list = generated_seqs['sequence'].tolist()
    train_list = train_seqs['sequence'].tolist()
    
    # Generate random pairs
    random.seed(42)  # for reproducibility
    pairs = []
    for _ in range(n_samples):
        g = random.choice(gen_list)
        t = random.choice(train_list)
        pairs.append((g, t))
    
    # Parallel processing
    chunk_size = max(1, len(pairs) // n_processes)
    chunks = [(pairs[i:i + chunk_size], k) for i in range(0, len(pairs), chunk_size)]
    
    with Pool(n_processes) as pool:
        results = pool.map(process_jaccard_chunk, chunks)
    
    distances = [d for sublist in results for d in sublist]
    return np.mean(distances)


def calculate_mpd_sampled(sequences, k=K_MER_SIZE, n_samples=100000, n_processes=None):
    """
    MPD: Sample random pairs instead of all combinations
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    seq_list = sequences['sequence'].tolist()
    n = len(seq_list)
    
    # Sample random unique pairs
    random.seed(42)
    pairs = set()
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(pairs) < n_samples and attempts < max_attempts:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j:
            pairs.add((i, j))
        attempts += 1
    
    # Convert indices to sequences
    pair_seqs = [(seq_list[i], seq_list[j]) for i, j in pairs]
    
    # Parallel processing
    chunk_size = max(1, len(pair_seqs) // n_processes)
    chunks = [(pair_seqs[i:i + chunk_size], k) for i in range(0, len(pair_seqs), chunk_size)]
    
    with Pool(n_processes) as pool:
        results = pool.map(process_jaccard_chunk, chunks)
    
    distances = [d for sublist in results for d in sublist]
    return np.mean(distances)


def calculate_uniqueness(generated_seqs, train_seqs):
    """Percentage of sequences not in training set"""
    train_set = set(train_seqs['sequence'])
    unique = sum(1 for seq in generated_seqs['sequence'] if seq not in train_set)
    return (unique / len(generated_seqs)) * 100


def kmer_distribution(sequences, k=K_MER_SIZE):
    """Get normalized k-mer distribution - MEMORY EFFICIENT"""
    counter = Counter()
    total = 0
    
    for seq in sequences['sequence']:
        kmers = [seq[i:i+k] for i in range(len(seq)-k+1)]
        counter.update(kmers)
        total += len(kmers)
    
    return {kmer: count/total for kmer, count in counter.items()}


def jsd(p, q):
    """Jensen-Shannon Divergence"""
    all_keys = set(p.keys()) | set(q.keys())
    p_vec = np.array([p.get(k, 0) for k in all_keys])
    q_vec = np.array([q.get(k, 0) for k in all_keys])
    
    epsilon = 1e-10
    p_vec = p_vec + epsilon
    q_vec = q_vec + epsilon
    p_vec = p_vec / p_vec.sum()
    q_vec = q_vec / q_vec.sum()
    
    m = 0.5 * (p_vec + q_vec)
    kl_p = np.sum(p_vec * np.log(p_vec / m))
    kl_q = np.sum(q_vec * np.log(q_vec / m))
    
    return 0.5 * (kl_p + kl_q)


def calculate_jsd_3mer(generated_seqs, train_seqs, k=K_MER_SIZE):
    """JSD for k-mer distributions"""
    p_gen = kmer_distribution(generated_seqs, k)
    p_train = kmer_distribution(train_seqs, k)
    return jsd(p_gen, p_train)


def calculate_jsd_gc(generated_seqs, train_seqs, bins=20):
    """JSD for GC-content distributions"""
    gc_gen = generated_seqs['GC_content'].values
    gc_train = train_seqs['GC_content'].values
    
    hist_gen, bin_edges = np.histogram(gc_gen, bins=bins, range=(0, 1), density=True)
    hist_train, _ = np.histogram(gc_train, bins=bin_edges, density=True)
    
    hist_gen = np.maximum(hist_gen, 1e-10)
    hist_train = np.maximum(hist_train, 1e-10)
    
    hist_gen = hist_gen / hist_gen.sum()
    hist_train = hist_train / hist_train.sum()
    
    m = 0.5 * (hist_gen + hist_train)
    kl_gen = np.sum(hist_gen * np.log(hist_gen / m))
    kl_train = np.sum(hist_train * np.log(hist_train / m))
    
    return 0.5 * (kl_gen + kl_train)


def calculate_jsd_entropy(generated_seqs, train_seqs, bins=20):
    """JSD for Shannon entropy distributions"""
    h_gen = generated_seqs['Shennon_entropy'].values
    h_train = train_seqs['Shennon_entropy'].values
    
    h_min = min(h_gen.min(), h_train.min())
    h_max = max(h_gen.max(), h_train.max())
    
    hist_gen, bin_edges = np.histogram(h_gen, bins=bins, range=(h_min, h_max), density=True)
    hist_train, _ = np.histogram(h_train, bins=bin_edges, density=True)
    
    hist_gen = np.maximum(hist_gen, 1e-10)
    hist_train = np.maximum(hist_train, 1e-10)
    
    hist_gen = hist_gen / hist_gen.sum()
    hist_train = hist_train / hist_train.sum()
    
    m = 0.5 * (hist_gen + hist_train)
    kl_gen = np.sum(hist_gen * np.log(hist_gen / m))
    kl_train = np.sum(hist_train * np.log(hist_train / m))
    
    return 0.5 * (kl_gen + kl_train)


# ==================== MAIN CALCULATION ====================

def calculate_all_metrics(generated_seqs, train_seqs, n_processes=None, n_samples=100000):
    """
    Calculate all metrics with sampling for large datasets
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Using {n_processes} processes")
    print(f"Generated sequences: {len(generated_seqs)}")
    print(f"Train sequences: {len(train_seqs)}")
    print(f"Sampling {n_samples} pairs for APJD/MPD")
    
    results = {}
    
    # 1. Uniqueness
    print("Calculating Uniqueness...")
    results['uniqueness'] = calculate_uniqueness(generated_seqs, train_seqs)
    
    # 2. APJD (sampled)
    print("Calculating APJD (sampled)...")
    results['apjd'] = calculate_apjd_sampled(
        generated_seqs, train_seqs, 
        n_samples=n_samples, n_processes=n_processes
    )
    
    # 3. MPD (sampled)
    print("Calculating MPD (sampled)...")
    results['mpd'] = calculate_mpd_sampled(
        generated_seqs, 
        n_samples=n_samples, n_processes=n_processes
    )
    
    # 4. JSD 3-mers
    print("Calculating JSD (3-mers)...")
    results['jsd_3mer'] = calculate_jsd_3mer(generated_seqs, train_seqs)
    
    # 5. JSD GC-content
    print("Calculating JSD (GC-content)...")
    results['jsd_gc'] = calculate_jsd_gc(generated_seqs, train_seqs)
    
    # 6. JSD Shannon entropy
    print("Calculating JSD (Shannon entropy)...")
    results['jsd_entropy'] = calculate_jsd_entropy(generated_seqs, train_seqs)
    
    return results


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    train_seqs = pd.read_csv('C:/Users/danya/Downloads/Train/distr+real.csv')
    generated_seqs = pd.read_csv('C:/Users/danya/Downloads/generated/wgan_distr_based_with_stats_tuned.csv')
    
    metrics = calculate_all_metrics(
        generated_seqs, train_seqs, 
        n_processes=12, 
        n_samples=100000000
    )
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric:20s}: {value:.4f}")