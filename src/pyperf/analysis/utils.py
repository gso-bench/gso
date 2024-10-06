import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-4")


def count_tokens(context: str):
    return len(tokenizer.encode(context, disallowed_special=()))
