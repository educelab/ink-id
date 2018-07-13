import random
import subprocess

for i in range(10):
    command = [
        'sbatch',
        '--array=2%2',
        'scripts/slurm_train_and_predict.sh',
        '/home/srpa226/data/dri-experiments-drive/2018-ml-inkid/data/CarbonPhantoms/CarbonSampleV3-June2017/CarbonPhantomV3.volpkg/working/2/Col2_k-fold-characters-region-set.json',  # NOQA
        '~/data/dri-experiments-drive/2018-ml-inkid/results/carbon_phantom/random_hyperparameters/',  # NOQA
        '--final-prediction-on-all',
        '--rclone-transfer-remote', 'dri-experiments-drive',
        '--training-max-batches', '100000',
        '--subvolume-shape', '28', '42', '42',
        # '--move-along-normal', str(random.randint(-14, 14)),
        # '--jitter-max', str(random.randint(0, 14)),
        '--learning-rate', str(random.random()),
        '--drop-rate', str(random.random()),
        '--batch-norm-momentum', str(random.random()),
    ]
    subprocess.call(command)
