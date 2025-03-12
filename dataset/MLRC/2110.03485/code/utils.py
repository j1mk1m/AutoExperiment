import os


def hparams_to_str(hparams):
    hparams_str = '_'.join([f'{key}{value}' for key, value in hparams.items()])
    return hparams_str


def get_files(logdir, l1lambda):
    files = [f for f in os.listdir(
        logdir) if f'CNX_l1lambda{l1lambda}_RDE_l1lambda{l1lambda}' in f]
    return files
