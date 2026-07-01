import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense, Dropout,
                                     Flatten, Input, LeakyReLU, Reshape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_FT_DATA = os.path.join(PROJECT_ROOT, 'Data', 'Sequence_Craft', 'SequenceCraft_dataset.csv')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Legacy Keras GAN fine-tuning (pure adversarial loss, no MFE/stop)'
    )
    parser.add_argument('--new-data-csv', default=DEFAULT_FT_DATA)
    parser.add_argument('--output-dir', default=os.path.join(PROJECT_ROOT, 'checkpoints', 'keras_ft'))
    parser.add_argument('--pretrained-path', default='',
                        help='Optional Keras .weights.h5 path (not compatible with WGAN pretrain)')
    parser.add_argument('--latent-dim', type=int, default=100)
    parser.add_argument('--max-seq-length', type=int, default=100)
    parser.add_argument('--vocab-size', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--sample-interval', type=int, default=100)
    parser.add_argument('--lr-g', type=float, default=2e-5)
    parser.add_argument('--lr-d', type=float, default=5e-5)
    return parser.parse_args()


def load_and_preprocess_data(csv_path, max_seq_length, vocab_size, charmap):
    df = pd.read_csv(csv_path)
    if 'e' in df.columns:
        sequences = df['e'].astype(str).tolist()
    elif 'Sequence' in df.columns:
        sequences = df['Sequence'].astype(str).tolist()
    elif 'sequence' in df.columns:
        sequences = df['sequence'].astype(str).tolist()
    else:
        raise KeyError("CSV must contain one of: 'e', 'Sequence', 'sequence'")

    X = np.zeros((len(sequences), max_seq_length, vocab_size), dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.upper()
        for t, char in enumerate(seq[:max_seq_length]):
            idx = charmap.get(char, charmap['A'])
            X[i, t, idx] = 1.0
    return X


def build_generator(latent_dim, max_seq_length, vocab_size):
    return Sequential([
        Input(shape=(latent_dim,)),
        Dense(256),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(max_seq_length * vocab_size),
        Reshape((max_seq_length, vocab_size)),
        tf.keras.layers.Softmax(axis=-1),
    ])


def build_discriminator(max_seq_length, vocab_size):
    return Sequential([
        Input(shape=(max_seq_length, vocab_size)),
        Conv1D(64, 5, strides=2, padding='same'),
        LeakyReLU(0.2),
        Dropout(0.25),
        Conv1D(128, 5, strides=2, padding='same'),
        LeakyReLU(0.2),
        Dropout(0.25),
        Flatten(),
        Dense(1, activation='sigmoid'),
    ])


def decode_sequence(one_hot_seq, rev_charmap):
    seq = []
    for pos in one_hot_seq:
        char_idx = np.argmax(pos)
        if char_idx == 4:
            break
        seq.append(rev_charmap[char_idx])
    return ''.join(seq)


def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)


def sample_sequences(generator, output_dir, epoch, latent_dim, rev_charmap, num_samples=5):
    noise = tf.random.normal((num_samples, latent_dim))
    gen_seqs = generator.predict(noise, verbose=0)

    plt.figure(figsize=(10, num_samples * 2))
    for i in range(num_samples):
        seq_str = decode_sequence(gen_seqs[i], rev_charmap)
        plt.text(
            0.5, (num_samples - i) / num_samples,
            f"Sample {i + 1}:\n{seq_str}\nLength: {len(seq_str)}",
            ha='center', va='center',
        )
        plt.axis('off')

    plt.savefig(os.path.join(output_dir, f'seqs_{epoch}.png'))
    plt.close()

    with open(os.path.join(output_dir, f'seqs_{epoch}.txt'), 'w') as f:
        for i in range(num_samples):
            seq_str = decode_sequence(gen_seqs[i], rev_charmap)
            f.write(f"Sample {i + 1}:\n{seq_str}\nLength: {len(seq_str)}\n\n")


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    charmap = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
    rev_charmap = {v: k for k, v in charmap.items()}

    X_train = load_and_preprocess_data(
        args.new_data_csv, args.max_seq_length, args.vocab_size, charmap
    )

    generator = build_generator(args.latent_dim, args.max_seq_length, args.vocab_size)
    discriminator = build_discriminator(args.max_seq_length, args.vocab_size)

    generator_optimizer = Adam(learning_rate=args.lr_g, beta_1=0.5)
    discriminator_optimizer = Adam(learning_rate=args.lr_d, beta_1=0.5)

    discriminator.compile(
        optimizer=discriminator_optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    if args.pretrained_path and os.path.exists(args.pretrained_path):
        generator.load_weights(args.pretrained_path)
        print(f"Loaded Keras weights: {args.pretrained_path}")

    d_loss = 0.0
    for epoch in range(args.epochs):
        noise = tf.random.normal((args.batch_size, args.latent_dim))
        gen_seqs = generator(noise, training=False)

        idx = np.random.randint(0, len(X_train), args.batch_size)
        real_seqs = X_train[idx]

        if epoch % 7 == 0:
            d_loss_real = discriminator.train_on_batch(
                real_seqs, np.ones((args.batch_size, 1))
            )
            d_loss_fake = discriminator.train_on_batch(
                gen_seqs, np.zeros((args.batch_size, 1))
            )
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        with tf.GradientTape() as tape:
            gen_seqs = generator(noise, training=True)
            validity = discriminator(gen_seqs)
            total_loss = generator_loss(validity)

        grads = tape.gradient(total_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % args.sample_interval == 0:
            d_val = float(d_loss[0]) if isinstance(d_loss, (list, np.ndarray)) else float(d_loss)
            g_val = float(total_loss.numpy()) if hasattr(total_loss, 'numpy') else float(total_loss)
            print(f"Epoch {epoch} | D Loss: {d_val:.4f} | G Loss: {g_val:.4f}")
            sample_sequences(generator, args.output_dir, epoch, args.latent_dim, rev_charmap)
            generator.save_weights(os.path.join(args.output_dir, f'generator_{epoch}.weights.h5'))
            discriminator.save_weights(os.path.join(args.output_dir, f'discriminator_{epoch}.weights.h5'))


if __name__ == '__main__':
    train(parse_args())
