import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.compat import v1 as tfv1

sys.path.append(os.getcwd())

import language_helpers
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import tflib.plot

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'Data')
DEFAULT_CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')


def parse_args():
    parser = argparse.ArgumentParser(description='WGAN-GP training / fine-tuning for DNA sequences')
    parser.add_argument('--dataset', choices=['eds', 'mfe', 'sequence_craft'], required=True,
                        help='Training dataset')
    parser.add_argument('--mode', choices=['pretrain', 'finetune'], default='pretrain',
                        help='pretrain: train from scratch; finetune: continue from checkpoint')
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                        help='Root directory with EDS/MFE/Sequence_Craft folders')
    parser.add_argument('--checkpoint-dir', default=None,
                        help='Directory to save checkpoints (default: checkpoints/<dataset> or checkpoints/<dataset>_ft)')
    parser.add_argument('--restore-checkpoint', default=None,
                        help='Checkpoint prefix to restore (required for finetune)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--iters', type=int, default=3600)
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--critic-iters', type=int, default=10)
    parser.add_argument('--lambda-gp', type=float, default=10.0)
    parser.add_argument('--max-n-examples', type=int, default=0,
                        help='Max sequences to load; 0 = use all data')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-4 for pretrain, 5e-5 for finetune)')
    parser.add_argument('--sample-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=200)
    return parser.parse_args()


def resolve_checkpoint_dir(args):
    if args.checkpoint_dir:
        return args.checkpoint_dir
    suffix = '_ft' if args.mode == 'finetune' else ''
    return os.path.join(DEFAULT_CHECKPOINT_ROOT, f"{args.dataset}{suffix}")


def softmax(logits, vocab_size):
    return tf.reshape(
        tf.nn.softmax(tf.reshape(logits, [-1, vocab_size])),
        tf.shape(logits)
    )


def make_noise(shape):
    return tf.random.normal(shape)


def build_res_block(name, dim, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name + '.1', dim, dim, 5, output)
    output = tf.nn.relu(output)
    output = lib.ops.conv1d.Conv1D(name + '.2', dim, dim, 5, output)
    return inputs + (0.3 * output)


def build_generator(n_samples, seq_len, dim, vocab_size):
    output = make_noise(shape=[n_samples, 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, seq_len * dim, output)
    output = tf.reshape(output, [-1, dim, seq_len])
    for i in range(1, 6):
        output = build_res_block(f'Generator.{i}', dim, output)
    output = lib.ops.conv1d.Conv1D('Generator.Output', dim, vocab_size, 1, output)
    output = tf.transpose(output, [0, 2, 1])
    return softmax(output, vocab_size)


def build_discriminator(inputs, seq_len, dim, vocab_size):
    output = tf.transpose(inputs, [0, 2, 1])
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', vocab_size, dim, 1, output)
    for i in range(1, 6):
        output = build_res_block(f'Discriminator.{i}', dim, output)
    output = tf.reshape(output, [-1, seq_len * dim])
    output = lib.ops.linear.Linear('Discriminator.Output', seq_len * dim, 1, output)
    return output


def inf_train_gen(lines, batch_size, charmap):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines) - batch_size + 1, batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i + batch_size]],
                dtype='int32'
            )


def decode_samples(samples, inv_charmap):
    decoded_samples = []
    for sample in samples:
        decoded = []
        for idx in sample:
            decoded.append(inv_charmap[idx])
        decoded_samples.append(tuple(decoded))
    return decoded_samples


def train(args):
    tfv1.disable_eager_execution()

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {len(gpus)} device(s)")
    else:
        print("Using CPU")

    checkpoint_dir = resolve_checkpoint_dir(args)
    samples_dir = os.path.join(checkpoint_dir, 'samples')
    plots_dir = os.path.join(checkpoint_dir, 'plots')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    tflib.plot.init(plots_dir)

    if args.mode == 'finetune' and not args.restore_checkpoint:
        raise ValueError('finetune mode requires --restore-checkpoint')

    learning_rate = args.lr if args.lr is not None else (5e-5 if args.mode == 'finetune' else 1e-4)

    settings = vars(args).copy()
    settings['checkpoint_dir'] = checkpoint_dir
    settings['learning_rate'] = learning_rate
    lib.print_model_settings(settings)

    lines, charmap, inv_charmap = language_helpers.load_dataset(
        max_length=args.seq_len,
        max_n_examples=args.max_n_examples if args.max_n_examples > 0 else None,
        dataset=args.dataset,
        data_root=args.data_root,
    )
    vocab_size = len(charmap)

    real_inputs_discrete = tfv1.placeholder(tf.int32, shape=[args.batch_size, args.seq_len])
    real_inputs = tf.one_hot(real_inputs_discrete, vocab_size)
    fake_inputs = build_generator(args.batch_size, args.seq_len, args.dim, vocab_size)
    fake_inputs_discrete = tf.argmax(fake_inputs, fake_inputs.get_shape().ndims - 1)

    disc_real = build_discriminator(real_inputs, args.seq_len, args.dim, vocab_size)
    disc_fake = build_discriminator(fake_inputs, args.seq_len, args.dim, vocab_size)

    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    with tf.device('/cpu:0'):
        alpha = tf.random.uniform(shape=[args.batch_size, 1, 1], minval=0., maxval=1.)
        differences = fake_inputs - real_inputs
        interpolates = real_inputs + (alpha * differences)
        gradients = tf.gradients(
            build_discriminator(interpolates, args.seq_len, args.dim, vocab_size),
            [interpolates],
        )[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += args.lambda_gp * gradient_penalty

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    gen_train_op = tfv1.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=0.5, beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)

    disc_train_op = tfv1.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=0.5, beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    ngram_sample_size = min(len(lines), 20000)
    ngram_lines = lines[:ngram_sample_size] if len(lines) <= ngram_sample_size else [
        lines[i] for i in np.random.choice(len(lines), ngram_sample_size, replace=False)
    ]
    print(f"Building n-gram models on {len(ngram_lines)} sequences (of {len(lines)} total)...")

    true_char_ngram_lms = [
        language_helpers.NgramLanguageModel(i + 1, ngram_lines[10 * args.batch_size:], tokenize=False)
        for i in range(4)
    ]
    validation_char_ngram_lms = [
        language_helpers.NgramLanguageModel(i + 1, ngram_lines[:10 * args.batch_size], tokenize=False)
        for i in range(4)
    ]
    for i in range(4):
        print("validation set JSD for n={}: {}".format(
            i + 1, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])
        ))
    true_char_ngram_lms = [
        language_helpers.NgramLanguageModel(i + 1, ngram_lines, tokenize=False) for i in range(4)
    ]

    session_config = tfv1.ConfigProto(allow_soft_placement=True)
    session_config.gpu_options.allow_growth = True
    session_config.graph_options.optimizer_options.global_jit_level = (
        tfv1.OptimizerOptions.OFF
    )

    with tfv1.Session(config=session_config) as session:
        saver = tfv1.train.Saver(max_to_keep=None)

        if args.restore_checkpoint:
            saver.restore(session, args.restore_checkpoint)
            print(f"Restored checkpoint: {args.restore_checkpoint}")
        else:
            session.run(tfv1.global_variables_initializer())

        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            return decode_samples(samples, inv_charmap)

        gen = inf_train_gen(lines, args.batch_size, charmap)

        for iteration in range(args.iters):
            start_time = time.time()

            if iteration % args.save_interval == 0 or iteration == args.iters - 1:
                saver.save(
                    session,
                    os.path.join(checkpoint_dir, 'model'),
                    global_step=iteration,
                    write_meta_graph=True,
                )
                print(f"Saved checkpoint at iteration {iteration}")

            if iteration > 0:
                session.run(gen_train_op)

            for _ in range(args.critic_iters):
                batch = next(gen)
                disc_cost_val, _ = session.run(
                    [disc_cost, disc_train_op],
                    feed_dict={real_inputs_discrete: batch},
                )

            tflib.plot.plot('time', time.time() - start_time)
            tflib.plot.plot('train disc cost', disc_cost_val)

            if iteration % args.sample_interval == args.sample_interval - 1:
                samples = []
                for _ in range(10):
                    samples.extend(generate_samples())

                for i in range(4):
                    lm = language_helpers.NgramLanguageModel(i + 1, samples, tokenize=False)
                    tflib.plot.plot('js{}'.format(i + 1), lm.js_with(true_char_ngram_lms[i]))

                sample_path = os.path.join(samples_dir, f'samples_{iteration}.txt')
                with open(sample_path, 'w') as f:
                    for s in samples:
                        f.write(''.join(s).replace('`', '') + '\n')

            if iteration % args.sample_interval == args.sample_interval - 1:
                tflib.plot.flush()

            tflib.plot.tick()

        tflib.plot.save_summary(title=f'{args.dataset} {args.mode}')


if __name__ == '__main__':
    train(parse_args())
