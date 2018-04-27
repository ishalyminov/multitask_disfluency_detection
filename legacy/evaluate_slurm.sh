#!/bin/bash

#SBATCH --partition=amd-longq
#SBATCH --nodes 1
#SBATCH --gres=gpu:k6000
#SBATCH --mail-user=is33@hw.ac.uk
#SBATCH --mail-type=ALL

# do some stuff to setup the environment
module purge
module load shared
module load cuda80/blas/8.0.61 cuda80/fft/8.0.61 cuda80/gdk/352.79 cuda80/nsight/8.0.61 cuda80/profiler/8.0.61 cuda80/toolkit/8.0.61 cudnn/6.0

# execute application (read in arguments from command line)
cd /home/ishalyminov/data/dialogue_denoiser && /home/ishalyminov/Envs/dialogue_denoiser/bin/python -m dialogue_denoiser \
  --evaluate \
  --data_dir ../babi_tools/dialogue_denoiser_data \
  --train_dir ckpt \
  --from_dev_data ../babi_tools/dialogue_denoiser_data_dstc6_3000/dialog-task1API-kb1_atmosphere-distr0.5-trn10000.txt/encoder.txt \
  --to_dev_data ../babi_tools/dialogue_denoiser_data_dstc6_3000/dialog-task1API-kb1_atmosphere-distr0.5-trn10000.txt/decoder.txt \
  --force_make_data
exit 0
