import re
    
def build_pubmedqa_messages(question: str, contexts: list[str]):
    """
    Build messages for PubMedQA task. (prompt engineering)
    """
    context_str = "\n\n".join(
        [f"Context {i + 1}:\n{c}" for i, c in enumerate(contexts)]
    )
    
    # Chinese prompt: 

    sys_content = (
        "你是一名资深的医学文献解读专家，下面是一道医学相关的判断题。"
        "根据给出的文献摘要内容与问题，判断问题的答案应为 yes、no 或 maybe 三者之一。\n"
        "请先进行必要的分析与推理，最后一行只输出最终结论，并严格使用如下格式：\n"
        "Final answer: yes\n"
        "或\n"
        "Final answer: no\n"
        "或\n"
        "Final answer: maybe\n"
        "请确保最后一行只有这一行内容，不要包含多余文字。\n"
    )
    user_content = (
        f"Question:\n{question}\n\n"
        f"Contexts:\n{context_str}\n\n"
        "请根据上述内容判断问题的答案是 yes、no 还是 maybe。"
    )
    
    
    # English prompt:
    
    # sys_content = (
    #     "You are an expert in interpreting medical literature. Below is a medical question. "
    #     "Based on the provided abstracts and the question, determine whether the answer is yes, no, or maybe.\n"
    #     "Please perform necessary analysis and reasoning first, and output the final conclusion in the following format on the last line only:\n"
    #     "Final answer: yes\n"
    #     "or\n"
    #     "Final answer: no\n"
    #     "or\n"
    #     "Final answer: maybe\n"
    #     "Ensure that the last line contains only this content without any extra text.\n"
    # )
    # user_content = (
    #     f"Question:\n{question}\n\n"
    #     f"Contexts:\n{context_str}\n\n"
    #     "Based on the above, determine whether the answer to the question is yes, no, or maybe."
    # )


    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def parse_pubmedqa_answer(response: str) -> str:
    """
    Parse the model response to extract the final decision (yes/no/maybe).
    Raises ValueError if parsing fails.
    """
    if response is None:
        raise ValueError("The model answers as empty.")

    text = response.strip()
    if not text:
        raise ValueError("The model answers as empty.")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = []

    strict_pattern = re.compile(
        r"^[#>*\s\*]*"
        r"(?:最终答案|答案|结论|Final answer|Answer|answer is|Assistant: Final answer|assistant: Final answer)"
        r"\s*[:：]?\s*"
        r"(yes|no|maybe)"
        r"\s*[*\s]*$",
        flags=re.IGNORECASE,
    )

    for line in reversed(lines):
        m = strict_pattern.search(line)
        if m:
            candidates.append(m.group(1).lower())
            break

    if not candidates:
        raise ValueError("No candidate decision (yes/no/maybe) can be parsed from the model's responses.")

    return candidates[0]