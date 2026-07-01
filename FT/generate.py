import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, LeakyReLU, BatchNormalization,
                                   Reshape, Input)
import datetime
from tqdm import tqdm

from mfe_utils import calculate_mfe

# Конфигурация ДОЛЖНА совпадать с обучением
config = {
    'latent_dim': 100,
    'max_seq_length': 100,
    'vocab_size': 5,  # A, T, C, G, N
    'min_seq_length': 20,
    # NUPACK DNA и ViennaRNA дают разные шкалы энергии — при необходимости перекалибруйте target_mfe
    'target_mfe': -10.0,
    'output_dir': ''
}

# Character mappings
charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
rev_charmap = {v: k for k, v in charmap.items()}

def build_generator():
    """Точная копия генератора из обучающего скрипта"""
    model = Sequential([
        Input(shape=(config['latent_dim'],)),
        Dense(256),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(config['max_seq_length'] * config['vocab_size']),
        Reshape((config['max_seq_length'], config['vocab_size'])),
        tf.keras.layers.Softmax(axis=-1)
    ])
    return model

def decode_sequence(one_hot_seq):
    """Декодирование с остановкой на N"""
    seq = []
    for pos in one_hot_seq:
        char_idx = np.argmax(pos)
        if char_idx == charmap['N']:
            break
        seq.append(rev_charmap[char_idx])
    return ''.join(seq)

def generate_high_quality_sequences(generator, num_sequences, min_mfe=-20.0, min_length=30):
    """Генерация последовательностей с фильтрацией по качеству"""
    sequences = []
    mfe_values = []
    lengths = []

    print(f"Генерация {num_sequences} высококачественных последовательностей (MFE: NUPACK DNA)...")

    with tqdm(total=num_sequences) as pbar:
        while len(sequences) < num_sequences:
            noise = tf.random.normal((min(1000, num_sequences - len(sequences)),
                                    config['latent_dim']))
            gen_seqs = generator.predict(noise, verbose=0)

            for seq in gen_seqs:
                seq_str = decode_sequence(seq)
                mfe = calculate_mfe(seq_str)
                length = len(seq_str)

                if (mfe <= min_mfe and
                    length >= min_length and
                    length <= config['max_seq_length']):

                    sequences.append(seq_str)
                    mfe_values.append(mfe)
                    lengths.append(length)
                    pbar.update(1)

                    if len(sequences) >= num_sequences:
                        break

    return sequences, mfe_values, lengths

def save_sequences(sequences, mfe_values, lengths, filename):
    """Сохранение результатов в CSV"""
    df = pd.DataFrame({
        'Sequence': sequences,
        'Length': lengths,
        'MFE': mfe_values
    })

    os.makedirs(config['output_dir'], exist_ok=True)
    filepath = os.path.join(config['output_dir'], filename)
    df.to_csv(filepath, index=False)
    print(f"Сохранено {len(sequences)} последовательностей в {filepath}")

    stats = {
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'mean_mfe': np.mean(mfe_values),
        'min_mfe': np.min(mfe_values),
        'max_mfe': np.max(mfe_values),
        'num_sequences': len(sequences)
    }

    stats_file = os.path.join(config['output_dir'], 'generation_stats.txt')
    with open(stats_file, 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

    return df

def load_generator(checkpoint_path):
    """Загрузка генератора из чекпоинта"""
    generator = build_generator()

    if checkpoint_path.endswith('.h5'):
        generator.load_weights(checkpoint_path)
    else:
        checkpoint = tf.train.Checkpoint(generator=generator)
        checkpoint.restore(checkpoint_path).expect_partial()

    print("Генератор успешно загружен")
    return generator

if __name__ == "__main__":
    checkpoint_path = ""

    generator = load_generator(checkpoint_path)

    num_sequences = 250000
    sequences, mfe_values, lengths = generate_high_quality_sequences(
        generator,
        num_sequences=num_sequences,
        min_mfe=config['target_mfe'],
        min_length=config['min_seq_length']
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_sequences_{timestamp}.csv"
    save_sequences(sequences, mfe_values, lengths, filename)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=30)
    plt.title("Распределение длин")
    plt.xlabel("Длина")
    plt.ylabel("Количество")

    plt.subplot(1, 2, 2)
    plt.hist(mfe_values, bins=30)
    plt.title("Распределение MFE (NUPACK DNA)")
    plt.xlabel("MFE")

    plt.tight_layout()
    plot_path = os.path.join(config['output_dir'], f"distributions_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"Графики распределений сохранены в {plot_path}")
