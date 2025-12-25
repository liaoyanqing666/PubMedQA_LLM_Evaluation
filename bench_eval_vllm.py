# Evaluation script for benchmark using vLLM
import os
import json
import traceback
from collections import Counter

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from bench_eval_messages import *

torch.backends.cuda.enable_flash_sdp(True)

def load_model_and_tokenizer(model_path: str,
                             visible_gpus: str,
                             dtype=torch.bfloat16,
                             lora_path: str | None = None,
                             max_tokens: int | None = None):
    """
    Load model and tokenizer.
    """
    gpu_list = [g.strip() for g in str(visible_gpus).split(",") if g.strip()]
    clean_visible = ",".join(gpu_list) if gpu_list else ""
    if clean_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = clean_visible

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='left',
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype == torch.bfloat16:
        vllm_dtype = "bfloat16"
    elif dtype == torch.float16:
        vllm_dtype = "float16"
    else:
        vllm_dtype = "auto"

    tp_size = max(len(gpu_list), 1)

    llm = LLM(
        model=model_path,
        tokenizer=lora_path if lora_path else model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        dtype=vllm_dtype,
        enable_lora=lora_path is not None,
        gpu_memory_utilization=0.90,
        max_model_len=max_tokens,
        max_lora_rank=64,
    )

    llm.name_or_path = model_path
    return llm, tokenizer


def generate_batch(model,
                   tokenizer,
                   conversations,
                   max_tokens: int = None,
                   lora_path: str | None = None,
                   vote_num: int = 1,
                   **kwargs):
    """
    Generate vote_num responses for all conversations.
    """
    texts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
        )
        for conv in conversations
    ]

    if max_tokens is None:
        engine_cfg = getattr(getattr(model, "llm_engine", None), "model_config", None)
        if engine_cfg is not None and hasattr(engine_cfg, "max_model_len"):
            max_tokens = engine_cfg.max_model_len
        else:
            max_tokens = 32768

    temperature = kwargs.pop('temperature', 0.0 if vote_num == 1 else 0.5)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
        skip_special_tokens=True,
        n=vote_num,
        **kwargs
    )

    lora_request = None
    if lora_path is not None:
        lora_request = LoRARequest(
            lora_name="default_lora",
            lora_int_id=1,
            lora_path=lora_path,
        )

    outputs = model.generate(
        texts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )

    # Multi-voting
    if vote_num > 1:
        all_responses = []
        all_finish_reasons = []
        for out in outputs:
            responses = []
            finish_reasons = []
            for sample in out.outputs:
                responses.append(sample.text.strip())
                finish_reasons.append(sample.finish_reason)
            all_responses.append(responses)
            all_finish_reasons.append(finish_reasons)
        return all_responses, all_finish_reasons
    
    # Single vote case
    responses = []
    finish_reasons = []
    for out in outputs:
        if out.outputs:
            text = out.outputs[0].text
            finish_reason = out.outputs[0].finish_reason
        else:
            text = ""
            finish_reason = None
        responses.append(text.strip())
        finish_reasons.append(finish_reason)
    return responses, finish_reasons


def evaluate_with_voting(responses, options, parse_fn):
    """
    Parse all output responses and obtain the voted answers.
    Only for multi-voting routes.
    """
    parsed_answers = []
    valid_answers = []
    
    for resp in responses:
        try:
            ans = parse_fn(resp, options)
            parsed_answers.append(ans)
            valid_answers.append(ans)
        except Exception:
            parsed_answers.append(None)
    
    if not valid_answers:
        raise ValueError("All responses failed to parse")
    
    counts = Counter(valid_answers)
    most_common = counts.most_common(1)[0][0]
    
    return most_common, parsed_answers

def eval_single_pubmedqa_json(path: str,
                              model,
                              tokenizer,
                              max_tokens: int | None = None,
                              print_errors: bool = True,
                              record_file: bool = False,
                              lora_path: str | None = None,
                              vote_num: int = 1,
                              **kwargs):
    """
    Evaluate PubMedQA dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data = []
    for pid, item in raw_data.items():
        data.append({
            "id": pid,
            "question": item["QUESTION"],
            "contexts": item["CONTEXTS"],
            "final_decision": item["final_decision"],
        })

    total = len(data)

    invalid_cnt = 0
    valid_cnt = 0
    correct_cnt = 0
    error_list = []
    error_details = []

    conversations = []
    for item in data:
        q = item["question"]
        ctx = item["contexts"]
        conversations.append(build_pubmedqa_messages(q, ctx))

    try:
        responses, finish_reasons = generate_batch(
            model=model,
            tokenizer=tokenizer,
            conversations=conversations,
            max_tokens=max_tokens,
            lora_path=lora_path,
            vote_num=vote_num,
            **kwargs
        )
    except Exception:
        tb_str = traceback.format_exc()
        for idx, item in enumerate(data):
            error_list.append(idx)
            error_entry = {
                "idx": idx,
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "traceback": tb_str,
            }
            
            if vote_num > 1:
                error_entry["responses"] = [None] * vote_num
                for i in range(1, vote_num + 1):
                    item[f"model_response_{i}"] = None
                    item[f"model_decision_{i}"] = None
            else:
                error_entry["response"] = None
                item["model_response"] = None
            
            item["model_decision"] = None
            error_details.append(error_entry)

        invalid_cnt = total
        valid_cnt = 0
    else:
        for idx, item in enumerate(data):
            # Multi-voting
            if vote_num > 1:
                item_responses = responses[idx]
                
                # Store individual responses
                for i, resp in enumerate(item_responses, 1):
                    item[f"model_response_{i}"] = resp
                
                try:
                    final_decision, parsed_decisions = evaluate_with_voting(
                        item_responses,
                        None,
                        lambda r, o: parse_pubmedqa_answer(r)
                    )
                    
                    # Store individual decisions
                    for i, dec in enumerate(parsed_decisions, 1):
                        item[f"model_decision_{i}"] = dec
                        
                    item["model_decision"] = final_decision
                    valid_cnt += 1
                    
                    if final_decision == item["final_decision"]:
                        correct_cnt += 1
                        
                except Exception as e:
                    invalid_cnt += 1
                    error_list.append(idx)
                    tb_str = traceback.format_exc()
                    error_details.append({
                        "idx": idx,
                        "id": item.get("id", ""),
                        "question": item.get("question", ""),
                        "responses": item_responses,
                        "traceback": tb_str,
                    })
                    
                    for i in range(1, vote_num + 1):
                        if f"model_decision_{i}" not in item:
                            item[f"model_decision_{i}"] = None
                    item["model_decision"] = None
                    
            # Single vote case
            else:
                response = responses[idx]
                item["model_response"] = response

                try:
                    pred = parse_pubmedqa_answer(response)
                    item["model_decision"] = pred
                    valid_cnt += 1
                    if pred == item["final_decision"]:
                        correct_cnt += 1
                except Exception:
                    invalid_cnt += 1
                    error_list.append(idx)
                    tb_str = traceback.format_exc()
                    error_details.append({
                        "idx": idx,
                        "id": item.get("id", ""),
                        "question": item.get("question", ""),
                        "response": response,
                        "traceback": tb_str,
                    })
                    item["model_decision"] = None
    
    # Save all model answers and decisions. (same directory as input file)
    if record_file:
        if lora_path is not None:
            raw_model_name = lora_path.rstrip("/")
        else:
            raw_model_name = getattr(model, "name_or_path", "")
            raw_model_name = str(raw_model_name).rstrip("/")
        if "/" in raw_model_name:
            model_name = raw_model_name.split("/")[-1]
        else:
            model_name = raw_model_name or "model"

        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path)
        root, ext = os.path.splitext(base_name)
        
        if vote_num > 1:
            out_name = f"{root}_eval_{model_name}_{vote_num}vote.jsonl"
        else:
            out_name = f"{root}_eval_{model_name}.jsonl"
            
        out_path = os.path.join(base_dir, out_name)

        with open(out_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved eval results to {out_path}")

    # Print error details.
    if error_details and print_errors:
        for err in error_details:
            print(f"id idx:\n{err['idx']}\n")
            print(f"PubMed ID:\n{err['id']}\n")
            print(f"Question:\n{err['question']}\n")
            if vote_num > 1:
                print(f"Model responses:\n{err.get('responses')}\n")
            else:
                print(f"Model response:\n{err.get('response')}\n")
            print(err["traceback"])

    # Print evaluation summary.
    acc = (correct_cnt / valid_cnt) if valid_cnt > 0 else 0.0
    print(
        f"file: {path}\n"
        f"total: {total}, invalid: {invalid_cnt}, valid: {valid_cnt}, "
        f"correct: {correct_cnt}, valid accuracy: {acc:.4f}, overall accuracy: {(correct_cnt / total):.4f}"
    )
    print(f"error_list = {error_list}")

def eval_pubmedqa(model_path: str,
                  data_path: str,
                  visible_gpus: str,
                  max_tokens: int | None = None,
                  print_errors: bool = False,
                  record_file: bool = False,
                  lora_path: str | None = None,
                  vote_num: int = 1,
                  **kwargs):
    """
    Entrance for evaluating PubMedQA dataset.
    
    model_path: Path to the base model. Also can be an online model name.
    data_path: Path to the PubMedQA JSON data file.
    visible_gpus: Comma-separated GPU device ids to use.
    max_tokens: Maximum tokens for generation. If None, use model's max length.
    print_errors: Whether to print detailed error information of every invalid case.
    record_file: Whether to save all model answers and decisions to a file.
    lora_path: Optional path to LoRA weights to apply during evaluation.
    vote_num: Number of answers generated for each question. Vote if there is more than 1.
    kwargs: Additional parameters passed to the vllm.SamplingParams: e.g. temperature, top_k, top_p. In particular, the temperature is set to 0/0.5 by default (depending on whether vote_num is 1 or not).
    """

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        visible_gpus=visible_gpus,
        dtype=torch.bfloat16,
        lora_path=lora_path,
        max_tokens=max_tokens,
    )

    eval_single_pubmedqa_json(
        path=data_path,
        model=model,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        print_errors=print_errors,
        record_file=record_file,
        lora_path=lora_path,
        vote_num=vote_num,
        **kwargs
    )



if __name__ == "__main__":
    eval_pubmedqa(
        model_path="/nfsdata2/Qwen/Qwen3-32B", # Replace with your local model path/online model name
        # lora_path="models/sft_Qwen3", # Optional: path to LoRA weights,
        data_path="dataset/PubMedQA/PQA-L/ori_pqal.json",
        visible_gpus="2, 3",
        max_tokens=32768, # When vote_num > 1, it is recommended to make it smaller, such as 16384.
        print_errors=False,
        record_file=False,
        vote_num=1, # Whether to use multi-response voting
        temperature=0.6, # Default: 0 if vote_num == 1 else 0.5
        # top_p=0.95, # Optional
        # top_k=20, # Optional
    )