import argparse
import glob
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GAN_SCRIPT = os.path.join(PROJECT_ROOT, 'improved_wgan_training', 'gan_language.py')
CHECKPOINT_ROOT = os.path.join(PROJECT_ROOT, 'checkpoints')

SCENARIOS = {
    'eds': {
        'description': 'Pretrain on EDS only',
        'steps': [('pretrain', 'eds')],
    },
    'mfe': {
        'description': 'Pretrain on MFE only',
        'steps': [('pretrain', 'mfe')],
    },
    'eds_ft': {
        'description': 'Pretrain on EDS, fine-tune on Sequence Craft',
        'steps': [('pretrain', 'eds'), ('finetune', 'sequence_craft', 'eds')],
    },
    'mfe_ft': {
        'description': 'Pretrain on MFE, fine-tune on Sequence Craft',
        'steps': [('pretrain', 'mfe'), ('finetune', 'sequence_craft', 'mfe')],
    },
}


def find_latest_checkpoint(checkpoint_dir):
    meta_files = glob.glob(os.path.join(checkpoint_dir, 'model-*.meta'))
    if not meta_files:
        plain_meta = os.path.join(checkpoint_dir, 'model.meta')
        if os.path.exists(plain_meta):
            return os.path.join(checkpoint_dir, 'model')
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def step_from_meta(path):
        base = os.path.basename(path).replace('.meta', '')
        if '-' in base:
            return int(base.rsplit('-', 1)[-1])
        return 0

    latest = max(meta_files, key=step_from_meta)
    return latest.replace('.meta', '')


def run_gan_cmd(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, cwd=os.path.join(PROJECT_ROOT, 'improved_wgan_training'))


def run_scenario(name, iters_pretrain=3600, iters_finetune=1000, max_n_examples=0, save_interval=200):
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Choose from: {list(SCENARIOS)}")

    scenario = SCENARIOS[name]
    print(f"=== Scenario: {name} — {scenario['description']} ===")

    for step in scenario['steps']:
        if step[0] == 'pretrain':
            dataset = step[1]
            checkpoint_dir = os.path.join(CHECKPOINT_ROOT, dataset)
            cmd = [
                sys.executable, GAN_SCRIPT,
                '--dataset', dataset,
                '--mode', 'pretrain',
                '--checkpoint-dir', checkpoint_dir,
                '--iters', str(iters_pretrain),
                '--max-n-examples', str(max_n_examples),
                '--save-interval', str(save_interval),
            ]
            run_gan_cmd(cmd)

        elif step[0] == 'finetune':
            ft_dataset = step[1]
            pretrain_source = step[2]
            pretrain_checkpoint_dir = os.path.join(CHECKPOINT_ROOT, pretrain_source)
            restore_ckpt = find_latest_checkpoint(pretrain_checkpoint_dir)
            ft_checkpoint_dir = os.path.join(CHECKPOINT_ROOT, f"{pretrain_source}_ft")
            cmd = [
                sys.executable, GAN_SCRIPT,
                '--dataset', ft_dataset,
                '--mode', 'finetune',
                '--checkpoint-dir', ft_checkpoint_dir,
                '--restore-checkpoint', restore_ckpt,
                '--iters', str(iters_finetune),
                '--max-n-examples', str(max_n_examples),
                '--save-interval', str(save_interval),
            ]
            run_gan_cmd(cmd)

    print(f"=== Scenario {name} completed ===")


def main():
    parser = argparse.ArgumentParser(description='Run DNA GAN training scenarios')
    parser.add_argument('--scenario', choices=list(SCENARIOS), required=True,
                        help='Training scenario to run')
    parser.add_argument('--iters-pretrain', type=int, default=3600)
    parser.add_argument('--iters-finetune', type=int, default=1000)
    parser.add_argument('--max-n-examples', type=int, default=0,
                        help='Max sequences; 0 = all data')
    parser.add_argument('--save-interval', type=int, default=200)
    args = parser.parse_args()

    run_scenario(
        args.scenario,
        iters_pretrain=args.iters_pretrain,
        iters_finetune=args.iters_finetune,
        max_n_examples=args.max_n_examples,
        save_interval=args.save_interval,
    )


if __name__ == '__main__':
    main()
