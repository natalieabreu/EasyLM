#! /bin/bash

# This is the example script to pretrain a 7B LLaMA model on a TPU v4-512 pod.
# These hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

#SBATCH -J soo-tayl
#SBATCH -o printouts/tayl-40x3.16m-150m/%A_%a.out
#SBATCH -e printouts/tayl-40x3.16m-150m/%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=1-24%6
#SBATCH -p kempner_h100
#SBATCH --account=kempner_barak_lab
#SBATCH --constraint="h100"
#SBATCH -t 3-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=1450G
#SBATCH -c 96

# Put your WANDB API key here to enable logging to wandb.
export PYTHONPATH="${PWD}:$PYTHONPATH"
export PYTHONPATH="${PYTHONPATH}:/n/holystore01/LABS/barak_lab/Users/nabreu/.mamba/envs/EasyLM/lib/python3.10/site-packages"
export PYTHONPATH="${PYTHONPATH}:/n/holystore01/LABS/barak_lab/Users/nabreu/.mamba/envs/EasyLM/bin/python"
export WANDB_API_KEY=
export JAX_TRACEBACK_FILTERING=off

'''
If you get the error:
jaxlib.xla_extension.XlaRuntimeError: INTERNAL: libdevice not found at ./libdevice.10.bc
Then set this `xla_gpu_cuda_data_dir` flag like below
Make sure nvvm present at this directory, otherwise install cuda toolkit
'''
# export XLA_FLAGS='--xla_gpu_cuda_data_dir=/n/holystore01/LABS/barak_lab/Users/nabreu/.mamba/envs/EasyLM/'

module load python/3.10.12-fasrc01;
mamba activate EasyLM;

python sweep_launcher.py \
    --program='EasyLM.models.llama.llama_train_taylor' \
    --mesh_dim='4,1,1' \
    --load_checkpoint='trainstate_params::/n/netscratch/kempner_barak_lab/Everyone/opt-soo/150m-checkpoint/62300467/streaming_train_state_1024' \
    --dtype='fp32' \
    --total_steps=60 \
    --log_freq=1 \
    --eval_freq=1 \
    --eval_steps=100 \
    --inner_loop_iter=1024 \
    --gradient_accumulation_steps=1 \
    --save_model_freq=1 \
    --save_milestone_freq=2500 \
    --load_llama_config='' \
    --update_llama_config='' \
    --llama.base_model='150M' \
    --llama.initializer_range=1.0 \
    --load_dataset_state='' \
    --taylor_order=1 \
    --lr_sched='global_cosine' \
    --inner_loop_lr='0.001&0.003&0.01' \
    --inner_loop_warmup='0&0.2' \
    --inner_loop_wd='0&0.01' \
    --inner_b1=0.9 \
    --inner_b2='0.99&0.999' \
    --inner_clip_gradient=1 \
    --weight_average=True \
    --weight_average_decay=0.999 \
    --tokenizer='google-t5/t5-base' \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.text_processor.add_bos_token=False \
    --train_dataset.huggingface_dataset.pretokenized_dataset_dir='/n/netscratch/kempner_barak_lab/Everyone/opt-soo/train' \
    --train_dataset.huggingface_dataset.tokens_count_at_start=134217728 \
    --train_dataset.huggingface_dataset.path='allenai/c4' \
    --train_dataset.huggingface_dataset.name='en' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset_batch_size=128 \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.text_processor.add_bos_token=False \
    --eval_dataset.huggingface_dataset.pretokenized_dataset_dir='/n/netscratch/kempner_barak_lab/Everyone/opt-soo/val' \
    --eval_dataset.huggingface_dataset.path='allenai/c4' \
    --eval_dataset.huggingface_dataset.name='en' \
    --eval_dataset.huggingface_dataset.split='validation' \
    --eval_dataset.huggingface_dataset.batch_size=128 \
    --checkpointer.save_optimizer_state=True \
    --wandb_project='EasyLM--opt-second-order-llama-150M' \
    --wandb_dir='/n/netscratch/kempner_barak_lab/Lab/nabreu/SOO-LM/experiment_output/open_llama_7b' \
    --output_dir='/n/netscratch/kempner_barak_lab/Lab/nabreu/SOO-LM/checkpoint/' \
    --notes='Taylor fo 150m bsz=40x3.16m 3x chinchilla' \
    --experiment_id=$SLURM_JOB_ID
|& tee $HOME/output.txt