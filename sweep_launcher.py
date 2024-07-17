import sys
import itertools
import os

def parse_sweep_arguments(args):
    sweep_flags = {}
    static_flags = []
    program = None
    
    for arg in args:
        if arg.startswith('--program='):
            program = arg.split('=')[1]
        elif '=' in arg and '&' in arg:
            key, values = arg.split('=')
            values = values.strip('\'"')
            values_list = values.split('&')
            sweep_flags[key] = values_list
        else:
            static_flags.append(arg)
    
    if not program:
        raise ValueError("Program not specified. Use --program=<program_name> to specify the program.")
    
    return program, sweep_flags, static_flags

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

def main():
    args = sys.argv[1:]
    job_index = int(os.getenv('SLURM_ARRAY_TASK_ID', '0')) - 1
    
    program, sweep_flags, static_flags = parse_sweep_arguments(args)

    combinations = generate_combinations(sweep_flags)
    config = select_configuration(combinations, job_index)
    
    final_args = static_flags + [f"{key}={value}" for key, value in config.items()]
    command = ["python", "-m", program] + final_args
    
    print("Running command:", " ".join(command))
    os.execvp(command[0], command)

if __name__ == "__main__":
    main()
