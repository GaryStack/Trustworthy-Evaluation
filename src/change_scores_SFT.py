import json
import os
import argparse
import random
import re

import numpy as np
import torch
import datasets
import evaluate


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from src.utils import seed_torch
from src.utils import topk_index
from src.activation_processor_SFT import ActivationContrasting
from src.eval.templates import create_prompt_with_tulu_chat_format
from src.eval.gsm.examplars import EXAMPLARS as GSM_EXAMPLARS

from src.eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    dynamic_import_function,
    load_hooked_lm_and_tokenizer
)

exact_match = evaluate.load("exact_match")

def layer_patch_hook(value, hook, neurons, patched_values):
        try:
            if not isinstance(patched_values, torch.Tensor):
                patched_values = torch.tensor(patched_values)
            patched_values = patched_values.to(value)
            value[..., neurons] = patched_values
        except Exception as e:
            print(f'Error in hook {hook}', e)
        return value
    

def generation_prompt_template(input_):
        return f'### Input:\n{input_}\n\n### Response:\n'

def dynamic_patch(args, use_base_model):
        """  compute change scores on given prompts

        Args:
            prompts: list of input prompt
            names_filter: the type of activations (e.g. mlp, attn, etc) to be cached
            token_type: the token position to be cached (full prompt, prompt last token, completion)
        Returns:
            change_scores: change scores of each neuron
            neuron_ranks: neurons ranked by change scores
            first_mean: mean activation of neurons from first peft model
            first_std: activation std of neurons from first peft model
            second_mean: mean activation of neurons from second peft model
            second_std: activation std of neurons from second peft model
        """ 
        random.seed(42)

        print("Loading data...")
        test_data = []
        # breakpoint()
        with open(os.path.join(args.data_dir, f"test.jsonl")) as fin:
            for line in fin:
                example = json.loads(line)
                test_data.append({
                    "question": example["question"],
                    "answer": example["answer"].split("####")[1].strip()
                })
            
        # some numbers are in the `x,xxx` format, and we want to remove the comma
        for example in test_data:
            example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
            assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

        if args.max_num_examples and len(test_data) > args.max_num_examples:
            test_data = random.sample(test_data, args.max_num_examples)

        # breakpoint()
        global GSM_EXAMPLARS
        if args.n_shot:
            if len(GSM_EXAMPLARS) > args.n_shot:
                GSM_EXAMPLARS = random.sample(GSM_EXAMPLARS, args.n_shot)
            demonstrations = []
            for example in GSM_EXAMPLARS:
                if args.no_cot:
                    demonstrations.append(
                        "Quesion: " + example["question"] + "\n" + "Answer: " + example["short_answer"]
                    )
                else:
                    demonstrations.append(
                        "Question: " + example["question"] + "\n" + "Answer: " + example["cot_answer"]
                    )
            prompt_prefix = "Answer the following questions.\n\n" + "\n\n".join(demonstrations) + "\n\n"
        else:
            prompt_prefix = "Answer the following question.\n\n"

        prompts = []
        # chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
        # for example in test_data:
        #     prompt = prompt_prefix + "Question: " + example["question"].strip()
        #     if args.use_chat_format:
        #         messages = [{"role": "user", "content": prompt}]
        #         prompt = chat_formatting_function(messages, add_bos=False)
        #         if prompt[-1] in ["\n", " "]:
        #             prompt += "Answer:"
        #         else:
        #             prompt += " Answer:"
        #     else:
        #         prompt += "\nAnswer:"
        #     prompts.append(prompt)
        
        chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
        for example in test_data:
            prompt = prompt_prefix + generation_prompt_template(example["question"].strip())
            # prompt = prompt_prefix + "Question: " + example["question"].strip()
            # if args.use_chat_format:
            #     messages = [{"role": "user", "content": prompt}]
            #     prompt = chat_formatting_function(messages, add_bos=False)
            #     if prompt[-1] in ["\n", " "]:
            #         prompt += "Answer:"
            #     else:
            #         prompt += " Answer:"
            # else:
            #     prompt += "\nAnswer:"
            prompts.append(prompt)

        if args.model_name_or_path:
            print("Loading model and tokenizer...")
            load_fn = load_hooked_lm_and_tokenizer if args.hooked else load_hf_lm_and_tokenizer
            if args.is_PEFT:
                model, tokenizer = load_fn(
                    model_name_or_path=args.model_name_or_path, 
                    tokenizer_name_or_path=args.tokenizer_name_or_path, 
                    load_in_8bit=args.load_in_8bit, 
                    device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                    use_fast_tokenizer=not args.use_slow_tokenizer,
                    peft_name_or_path=args.second_peft_path
                )
            else :
                if use_base_model:
                    model, tokenizer = load_fn(
                        model_name_or_path=args.model_name_or_path, 
                        tokenizer_name_or_path=args.tokenizer_name_or_path, 
                        load_in_8bit=args.load_in_8bit, 
                        device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                        use_fast_tokenizer=not args.use_slow_tokenizer,
                    )
                else :
                    model, tokenizer = load_fn(
                        model_name_or_path=args.second_model_name_or_path, 
                        tokenizer_name_or_path=args.tokenizer_name_or_path, 
                        load_in_8bit=args.load_in_8bit, 
                        device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                        use_fast_tokenizer=not args.use_slow_tokenizer,
                    )
                    
            # breakpoint()    
            if args.first_model_name_or_path:
                if args.is_PEFT:
                    guided_model, _ = load_fn(
                        model_name_or_path=args.model_name_or_path, 
                        tokenizer_name_or_path=args.tokenizer_name_or_path, 
                        load_in_8bit=args.load_in_8bit, 
                        device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                        use_fast_tokenizer=not args.use_slow_tokenizer,
                        peft_name_or_path=args.first_peft_path
                    )
                else :
                    guided_model, _ = load_fn(
                        model_name_or_path=args.first_model_name_or_path, 
                        tokenizer_name_or_path=args.tokenizer_name_or_path, 
                        load_in_8bit=args.load_in_8bit, 
                        device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                        use_fast_tokenizer=not args.use_slow_tokenizer,
                    )
                
            else:
                guided_model = None
            causality_scores = torch.zeros(31, 14336)
            # i_values = np.arange(32)
            # j_values = np.arange(11008)
            # vector = np.array([i_values, j_values]).T
            # breakpoint()
            new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1] # get the last token because the tokenizer may add space tokens at the start.
            not_patching_outputs = generate_completions(
                                    model=model,
                                    tokenizer=tokenizer,
                                    prompts=prompts,
                                    max_new_tokens=256,
                                    batch_size=args.eval_batch_size,
                                    stop_id_sequences=[[tokenizer.eos_token_id]],#new_line_token
                                    do_sample=False,
                                    guided_model=None,
                                    index=None,
                                    hook_fn=layer_patch_hook
                                )
            for i in range(32):
                for j in range(0, 14336, 512): # 11008 llama2-7b 256
                    row = [[i, j + k] for k in range(512)]
                    topk_index = (torch.tensor(row))
                    
                    # breakpoint()
                    
                    outputs = generate_completions(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=prompts,
                        max_new_tokens=256,
                        batch_size=25,
                        stop_id_sequences=[[tokenizer.eos_token_id]],#new_line_token
                        do_sample=False,
                        guided_model=guided_model,
                        index=topk_index,
                        hook_fn=layer_patch_hook
                    )
                    not_patching_predictions = []
                    for output in not_patching_outputs:
                        # replace numbers like `x,xxx` with `xxxx`
                        output = re.sub(r"(\d),(\d)", r"\1\2", output)
                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
                        if numbers:
                            not_patching_predictions.append(numbers[-1])
                        else:
                            not_patching_predictions.append(output)

                    predictions = []
                    for output in outputs:
                        # replace numbers like `x,xxx` with `xxxx`
                        output = re.sub(r"(\d),(\d)", r"\1\2", output)
                        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
                        if numbers:
                            predictions.append(numbers[-1])
                        else:
                            predictions.append(output)
                        
                    # print("Calculating accuracy of patching model...")
                    targets = [example["answer"] for example in test_data]

                    not_patching_em_score = exact_match.compute(predictions=not_patching_predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
                    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
                    # print(f"Exact match of contaminated model : {not_patching_em_score}")
                    # print(f"Exact match of patching contaminated model : {em_score}")
                    if use_base_model:
                        for k in range(512): # llama 256
                            causality_scores[i,j+k] -= not_patching_em_score - em_score
                    else :
                        for k in range(512): # llama 256
                            causality_scores[i,j+k] += not_patching_em_score - em_score
                        
                    # breakpoint()
        else:
            instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
            results = query_openai_chat_model(
                engine=args.openai_engine,
                instances=instances,
                batch_size=args.eval_batch_size if args.eval_batch_size else 10,
                # output_path=os.path.join(args.save_dir, f"openai_results.jsonl"),
            )
            outputs = [result["output"] for result in results]

        return causality_scores

def main(args):
    
    seed_torch(42)
    if 'gsm' or 'math' in args.dataset:
        dataset = datasets.load_dataset('csv', data_files=args.dataset)
        # breakpoint()
        eval_data = dataset["train"]["question"]
    else :
        eval_data = datasets.load_dataset('json', data_files=args.dataset)["train"]["prompt"]
    if args.num_samples > 0:
        eval_data = eval_data[:args.num_samples]
        
    prompts = []
    # breakpoint()
    for example in eval_data:
        prompt = example.strip() 
        messages = [{"role": "user", "content": prompt}]
        prompt = create_prompt_with_tulu_chat_format(messages, add_bos=False)
        prompts.append(prompt+args.generation_startswith)
    # breakpoint()
    names_filter = lambda name: name.endswith('hook_post') and '31' not in name
    ac = ActivationContrasting(
        args.model_name_or_path,
        args.first_model_name_or_path,
        args.second_model_name_or_path,
        batchsize=args.eval_batch_size,
        max_new_tokens=args.max_new_tokens,
        device_map='balanced_low_0'
    )
    
    change_scores, first_mean, first_std, second_mean, second_std = ac.compute_change_scores(prompts, names_filter, args.token_type)
    
    base_causality_scores = dynamic_patch(args=args, use_base_model=True)
    contaminated_causality_scores = dynamic_patch(args=args, use_base_model=False)
    # breakpoint()
    final_locate_score = change_scores + base_causality_scores + contaminated_causality_scores
    # breakpoint()
    neuron_ranks = torch.cat([torch.tensor((i, j)).unsqueeze(0) for i, j in topk_index(final_locate_score, -1)], dim=0)
        
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    torch.save((final_locate_score.cpu(), neuron_ranks.cpu(), first_mean.cpu(), first_std.cpu(), second_mean.cpu(), second_std.cpu()), args.output_file)
    # breakpoint()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute change scores via generation-time activation contrasting.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens in generation.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--output_file",
        type=str, 
        default="../data/default.pt"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--first_model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--second_model_name_or_path", 
        # nargs='+', 
        type=str,
        default=None, 
        help="The folder contains peft checkpoint saved with PeftModel.save_pretrained()."
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--token_type", 
        type=str, 
        default='completion', 
        choices=['prompt', 'prompt_last', 'completion'],
        help="Compute change scores from which token position."
    )
    parser.add_argument(
        "--generation_startswith", 
        type=str, 
        default='', 
        help="Generation start with given prefix."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--hooked", 
        action="store_true", 
        help="If given, we're evaluating a hooked model."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="src.eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--no_cot", 
        action="store_true", 
        help="If given, we're evaluating a model without chain-of-thought."
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/gsm"
    )
    parser.add_argument(
        "--n_shot", 
        type=int, 
        default=8, 
        help="max number of examples to use for demonstration."
    )
    parser.add_argument(
        "--is_PEFT", 
        action="store_true", 
        help="If given, it is assumed that the model to be tested is fine-tuned by PEFT, otherwise it is fine-tuned by all parameters."
    )
    args = parser.parse_args()
    print(f"args.model_name_or_path is : {args.model_name_or_path}")
    print(f"args.first_model_name_or_path is : {args.first_model_name_or_path}")
    print(f"args.second_model_name_or_path is : {args.second_model_name_or_path}")
    # breakpoint()
    main(args)

