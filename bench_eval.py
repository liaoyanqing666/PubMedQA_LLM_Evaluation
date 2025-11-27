# Evaluation script for benchmark using transformers
import os
import json
import traceback
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from bench_eval_messages import *

torch.backends.cuda.enable_flash_sdp(True)

def load_model_and_tokenizer(model_path: str,
                             visible_gpus: str,
                             dtype=torch.bfloat16):
    """
    Load model and tokenizer.
    """
    
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side='left',
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2", # "sdpa", 
    )
    model.eval()
    return model, tokenizer


def generate_batch(model,
                   tokenizer,
                   conversations,
                   max_tokens: int | None = None):
    """
    Generate responses for a batch of conversations.
    """
    texts = [
        tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
        )
        for conv in conversations
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    generate_kwargs = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": False,
    }
    if max_tokens is not None:
        generate_kwargs["max_new_tokens"] = max_tokens
    else:
        generate_kwargs["max_length"] = model.config.max_position_embeddings

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs
        )

    gen_ids = outputs[:, input_ids.shape[1]:]
    responses = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    responses = [r.strip() for r in responses]
    return responses


def eval_single_pubmedqa_json(path: str,
                              model,
                              tokenizer,
                              batch_size: int = 8,
                              max_tokens: int | None = None,
                              print_errors: bool = True,
                              record_file: bool = False):
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

    pbar = tqdm(total=total, desc=f"Evaluate PubMedQA {os.path.basename(path)}")
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_items = data[start:end]

        conversations = []
        for item in batch_items:
            q = item["question"]
            ctx = item["contexts"]
            conversations.append(build_pubmedqa_messages(q, ctx))

        try:
            # Generate model responses for the batch.
            responses = generate_batch(
                model=model,
                tokenizer=tokenizer,
                conversations=conversations,
                max_tokens=max_tokens,
            )
        except Exception:
            tb_str = traceback.format_exc()
            for i, item in enumerate(batch_items):
                idx = start + i
                error_list.append(idx)
                error_details.append({
                    "idx": idx,
                    "id": item.get("id", ""),
                    "question": item.get("question", ""),
                    "response": None,
                    "traceback": tb_str,
                })
                data[idx]["model_response"] = None
                data[idx]["model_decision"] = None
            invalid_cnt += len(batch_items)
            pbar.update(len(batch_items))
            continue

        # Process model responses for the batch.
        for i, item in enumerate(batch_items):
            idx = start + i
            response = responses[i]
            data[idx]["model_response"] = response

            try:
                pred = parse_pubmedqa_answer(response)
                data[idx]["model_decision"] = pred
                valid_cnt += 1
                if pred == data[idx]["final_decision"]:
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
                data[idx]["model_decision"] = None

            pbar.update(1)

    pbar.close()

    # Save all model answers and decisions. (same directory as input file)
    if record_file:
        raw_model_name = getattr(model, "name_or_path", "")
        raw_model_name = raw_model_name.rstrip("/")
        if "/" in raw_model_name:
            model_name = raw_model_name.split("/")[-1]
        else:
            model_name = raw_model_name or "model"

        base_dir = os.path.dirname(path)
        base_name = os.path.basename(path)
        root, ext = os.path.splitext(base_name)
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
            print(f"Model response:\n{err['response']}\n")
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
                  batch_size: int = 8,
                  max_tokens: int | None = None,
                  print_errors: bool = True,
                  record_file: bool = False):
    """
    Entrance for evaluating PubMedQA dataset.
    
    model_path: Path to the base model. Also can be an online model name.
    data_path: Path to the PubMedQA JSON data file.
    visible_gpus: Comma-separated GPU device ids to use.
    batch_size: Evaluation batch size.
    max_tokens: Maximum tokens for generation. If None, use model's max length.
    print_errors: Whether to print detailed error information of every invalid case.
    record_file: Whether to save all model answers and decisions to a file.
    """
    
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        visible_gpus=visible_gpus,
        dtype=torch.bfloat16,
    )

    eval_single_pubmedqa_json(
        path=data_path,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_tokens=max_tokens,
        print_errors=print_errors,
        record_file=record_file,
    )


if __name__ == "__main__":
    eval_pubmedqa(
        model_path="/nfsdata4/Qwen/Qwen3-32B", # Replace with your local model path/online model name
        data_path="dataset/PubMedQA/PQA-L/ori_pqal.json",
        visible_gpus="2",
        batch_size=4,
        print_errors=False,
        record_file=False,
        max_tokens=32768,
    )