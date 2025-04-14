import os
import re
import argparse
import multiprocessing as mp
from math_eval import setup, set_seed, parse_args
from transformers import AutoConfig, AutoTokenizer
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


TOKENIZERS_PARALLELISM=False

def make(checkpoint_folder, tag, config, tokenizer, processed_list):
    print(f'>>>> tag {tag}')
    if tag == 'global_step0':
        print('global_step0 --> no action')
        return  # tag0 is the base model
    model_dir = os.path.join(checkpoint_folder, tag, 'hf_model')
    if os.path.exists(model_dir):
        print(f'!!! Path `{tag}/hf_model` already exists.')
        return
    os.makedirs(model_dir, exist_ok=True)

    # Save config and tokenizer
    config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Convert the checkpoint
    zero_checkpoint_path = os.path.join(model_dir, 'pytorch_model.bin')
    convert_zero_checkpoint_to_fp32_state_dict(checkpoint_folder, zero_checkpoint_path, tag=tag)
    processed_list.append(tag)


def HF_worker(job_queue, checkpoint_folder, config, tokenizer, processed_list):
    while not job_queue.empty():
        try:
            job_id, tag = job_queue.get_nowait()  # Get next job
        except:
            break  # Exit if queue is empty
        make(checkpoint_folder, tag, config, tokenizer, processed_list)


def convert_to_HF(checkpoint_folder, pretrain, num_processes):
    config = AutoConfig.from_pretrained(pretrain)
    tokenizer = AutoTokenizer.from_pretrained(pretrain)
    job_queue = mp.Queue()
    manager = mp.Manager()
    processed_list = manager.list()  # Shared list to track processed paths
    tags = [item for item in os.listdir(checkpoint_folder) if 'global_step' in item]
    tags = sorted(tags, key=lambda x: int(re.search(r'\d+$', x).group()))  # making sort incremental
    print(f'>> {len(tags)} checkpoints are found.')

    # Populate job queue with (job_id, input_path, output_path)
    for job_id, tag in enumerate(tags):
        job_queue.put((job_id, tag))

    # Create and start GPU workers
    processes = []
    for _ in range(num_processes):
        p = mp.Process(target=HF_worker, args=(job_queue, checkpoint_folder, config, tokenizer, processed_list))
        processes.append(p)
        p.start()

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print("Processing finished.")
    processed_list = list(processed_list)
    missing_paths = set(tags) - set(processed_list)
    print(f'\n {len(processed_list)} jobs completed. Missing Paths: ')
    for item in missing_paths:
        print(item)


def run_eval_job(job_id, input_path, output_path, gpu_id, data_names, eval_args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Job{job_id}|GPU{gpu_id} (Input: ...{input_path[-20:]}, Output: ...{output_path[-20:]})")

    new_args = {
        'model_name_or_path': input_path,
        'data_names': data_names,
        'output_dir': output_path,
        'start': 0,
        'end': -1,
        'use_vllm': True,
        'save_outputs': True,
        'pipeline_parallel_size': 1,
    }
    new_args.update(eval_args)

    for d in data_names.split(','):
        if d in ["aqua", "sat_math", "mmlu_stem"]:  # fix bug here too
            new_args.update({'overwrite': True, 'num_shots': 5})
            break

    args = parse_args()
    args_dict = vars(args)
    args_dict.update(new_args)
    args = argparse.Namespace(**args_dict)
    set_seed(args.seed)
    setup(args)



def eval_worker(job_queue, gpu_id, data_names, eval_args):
    while not job_queue.empty():
        try:
            job_id, input_path, output_path = job_queue.get_nowait()
        except:
            break
        run_eval_job(job_id, input_path, output_path, gpu_id, data_names, eval_args)

def evaluate(checkpoint_folder, num_gpus, pretrain, data_names, eval_args):
    job_queue = mp.Queue()
    tags = [item for item in os.listdir(checkpoint_folder) if 'global_step' in item]
    tags = sorted(tags, key=lambda x: int(re.search(r'\d+$', x).group()))  # making sort incremental
    print(f'>> {len(tags)} checkpoints are found.')
    
    # base model eval (init point)
    input_path = pretrain  # --> base model
    output_path = os.path.join(checkpoint_folder, 'global_step0', 'eval')
    os.makedirs(output_path, exist_ok=True)
    job_queue.put((0, input_path, output_path))

    # Populate job queue with (job_id, input_path, output_path)
    for job_id, tag in enumerate(tags):
        input_path = os.path.join(checkpoint_folder, tag, 'hf_model')
        output_path = os.path.join(checkpoint_folder, tag, 'eval')
        # if os.path.exists(output_path):
        #     continue
        job_queue.put((job_id+1, input_path, output_path))  # --> starting from 1 (job 0 is the pretrain model)

    # Create and start GPU workers
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=eval_worker, args=(job_queue, gpu_id, data_names, eval_args))
        processes.append(p)
        p.start()

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print("All jobs completed.")


def parse_main_args():
    parser = argparse.ArgumentParser()

    # Defaults set here
    parser.add_argument('--checkpoint_folder', type=str, default='../train/checkpoints/R1_on_math/_actor')
    parser.add_argument('--pretrain', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', choices=['deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen2.5-Math-1.5B'])

    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--num_processes', type=int, default=30)
    parser.add_argument('--data_names', type=str, default='math500,aime24')

    parser.add_argument('--prompt_type', type=str, default="r1-cot", choices=["r1-cot", "qwen25-math-cot"])
    parser.add_argument('--apply_chat_template', action='store_true')
    parser.add_argument('--n_sampling', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens_per_call', type=int, default=32768, choices=[32768, 3000])
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--num_test_sample', type=int, default=-1)

    return parser.parse_args()


def main_entry():
    args = parse_main_args()

    # === convert to huggingface ===
    convert_to_HF(args.checkpoint_folder, args.pretrain, args.num_processes)

    # === evaluate ===
    eval_args = {
        'prompt_type': args.prompt_type,
        'apply_chat_template': args.apply_chat_template,
        'n_sampling': args.n_sampling,
        'temperature': args.temperature,
        'max_tokens_per_call': args.max_tokens_per_call,
        'top_p': args.top_p,
        'split': args.split,
        'num_test_sample': args.num_test_sample
    }

    evaluate(args.checkpoint_folder, args.num_gpus, args.pretrain, args.data_names, eval_args)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main_entry()