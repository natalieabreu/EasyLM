import sys
import itertools
import os
import re

def parse_sweep_arguments(args):
    sweep_flags = {}
    static_flags = []
    program = None
    resume_from=None
    
    for arg in args:
        if arg.startswith('--program='):
            program = arg.split('=')[1]
        elif arg.startswith('--resume_from='):
            resume_from = arg.split('=')[1]
        elif '=' in arg and '&' in arg:
            key, values = arg.split('=')
            values = values.strip('\'"')
            values_list = values.split('&')
            sweep_flags[key] = values_list
        else:
            static_flags.append(arg)
    
    if not program:
        raise ValueError("Program not specified. Use --program=<program_name> to specify the program.")
    
    return program, resume_from, sweep_flags, static_flags

def generate_combinations(sweep_flags):
    if not sweep_flags:
        return [{}]
    keys, values = zip(*sweep_flags.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations

def select_configuration(combinations, job_index):
    if job_index >= len(combinations):
        raise ValueError("Job index out of range")
    return combinations[job_index]

# def format_value(value):
#     try:
#         int_value = int(value)
#         return str(int_value)  # Return integer values as-is
#     except ValueError:
#         return f"'{value}'"  # Wrap string values in quotation marks

def set_static_flag(static_flags, flag, value):
    static_flags = [arg for arg in static_flags if not arg.startswith(f'--{flag}=')]
    static_flags.append(f"--{flag}={value}")
    return static_flags

def get_latest_checkpoint(directory_path):
    """
    Returns the full path to the streaming_train_state file with the highest numerical suffix.

    Parameters:
    - directory_path (str): The path to the directory to search.

    Returns:
    - str or None: Full path to the latest streaming_train_state file, or None if not found.
    """
    try:
        # List all files in the directory
        files = os.listdir(directory_path)
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")
        return None
    except PermissionError:
        print(f"Permission denied to access: {directory_path}")
        return None

    # Regex pattern to match 'streaming_train_state_<number>'
    pattern = re.compile(r'^streaming_train_state_(\d+)$')

    streaming_files = []
    for file in files:
        match = pattern.match(file)
        if match:
            number = int(match.group(1))
            streaming_files.append((number, file))

    if not streaming_files:
        print("No streaming_train_state files found.")
        return None

    # Find the file with the highest numerical suffix
    max_number, max_file = max(streaming_files, key=lambda x: x[0])

    # Return the full path to the file
    return max_number, os.path.join(directory_path, max_file)

def main():
    args = sys.argv[1:]

    program, resume_from, sweep_flags, static_flags = parse_sweep_arguments(args)

    if resume_from:
        job_id = resume_from
        print(f"Resuming from job_id: {job_id}")
    else:
        job_id = os.getenv('SLURM_ARRAY_JOB_ID', '0')

    # Assumes index will be the same if resuming
    job_index = int(os.getenv('SLURM_ARRAY_TASK_ID', '0')) - 1
    job_id = f"{job_id}_{job_index}"

    static_flags = set_static_flag(static_flags, 'experiment_id', job_id)    

    combinations = generate_combinations(sweep_flags)
    config = select_configuration(combinations, job_index)

    train_dataset_batch_size = config.get('train_dataset_batch_size') or next((arg.split('=')[1] for arg in static_flags if arg.startswith('--train_dataset_batch_size=')), None)

    if not train_dataset_batch_size:
        raise ValueError("train_dataset_batch_size not specified. Use --train_dataset_batch_size=<batch_size> to specify the batch size.")

    print(train_dataset_batch_size)

    # check if logger.output_dir + job_id exists and if so, set load_checkpoint to "trainstate::logger.output_dir + job_id"
    logger_output_dir = next((arg.split('=')[1] for arg in static_flags if arg.startswith('--output_dir=')), None)

    if logger_output_dir:
        checkpoint_path = os.path.join(logger_output_dir, job_id)
        if os.path.exists(checkpoint_path):
            # step, ckpt = get_latest_checkpoint(checkpoint_path)
            ckpt = os.path.join(checkpoint_path, "streaming_train_state")

            if os.path.exists(ckpt):
                print(f"Resuming from path: {ckpt}", flush=True)
                static_flags = set_static_flag(static_flags, 'load_checkpoint', f"trainstate::{ckpt}")

                if os.path.exists(os.path.join(checkpoint_path, "streaming_ema_params")):
                    print("Loading ema checkpoint")
                    static_flags = set_static_flag(static_flags, 'load_ema_checkpoint', f"params::{os.path.join(checkpoint_path, 'streaming_ema_params')}")

                with open(f"{checkpoint_path}/wandb_id.txt", "r") as f:
                    wandb_id = f.read().strip()
                    print(f"Resuming from wandb_id: {wandb_id}")
                    static_flags = set_static_flag(static_flags, 'wandb_run_id', wandb_id)
        

    print(static_flags)
    
    final_args = static_flags + [f"{key}={value}" for key, value in config.items()] + [f"--train_dataset.huggingface_dataset.batch_size={train_dataset_batch_size}"]

    print(final_args)
    
    command = ["python", "-m", program] + final_args
    
    print("Running command:", " ".join(command))
    os.execvp(command[0], command)

if __name__ == "__main__":
    main()
