#!/bin/bash
#SBATCH -J funcodec_inference_one
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -p kshdnormal02
#SBATCH -o inference/log_infer_one.out
#SBATCH -e inference/log_infer_one.err

export MIOPEN_FIND_MODE=3
export HSA_FORCE_FINE_GRAIN_PRICE=1
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0

# export ROCBLAS_TENSILE_LIBPATH=/public/software/compiler/rocm/dtk-23.10/lib/rocblas/library_dcu2

source ~/anaconda3/etc/profile.d/conda.sh
conda activate bltang_new

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.7.4/gcc-7.3.1
module load compiler/rocm/dtk-23.04.1


python -u inference/infer.py --config_file config/conf_right.yaml --model_file /public/home/qinxy/bltang/laura_gpt/ckpt/conf_right/best.pth --output_dir /public/home/qinxy/bltang/laura_gpt/output_one --default_config inference/infer_one.yaml --raw_inputs "nothing was to be done but to put about, and return in disappointment towards the north." "started dating my boyfriend" "/public/home/qinxy/bltang/FunCodec/egs/LibriTTS/text2speech_laura/demo/jieni.wav" --tokenize_to_phone 