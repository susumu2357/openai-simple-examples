import argparse
import json
import os
from typing import Dict, List

import requests
from embeddings import save_json
from prompt import articles_relevance_order, compose_prompt


def call_gpt(messages: List[Dict[str, str]], temperature: float):
    url = "https://api.openai.com/v1/chat/completions"
    key = os.environ["OPENAI_API_KEY"]
    headers = {"Authorization": f"Bearer {key}"}
    data = {
        "messages": messages,
        "temperature": temperature,
        "model": "gpt-3.5-turbo",
        "stream": True
    }
    res = requests.post(url, headers=headers, json=data, stream=True)

    # Reference https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
    collected_messages = []
    for line in res.iter_lines(decode_unicode=True):
        # line is a bytes object
        # b'data: {
        # "id":"chatcmpl-7HD38WE6otovhZ21h0A4hyaArrAO1",
        # "object":"chat.completion.chunk","created":1684336094,
        # "model":"gpt-3.5-turbo-0301",
        # "choices":[
        # {"delta":{"role":"assistant"},"index":0,"finish_reason":null}
        # ]}'
        try:
            chunk = json.loads(line.replace("data: ", ""))
        except Exception:
            continue
        chunk_message = chunk["choices"][0]["delta"]
        collected_messages.append(chunk_message)
        if "content" in chunk_message.keys():
            print(chunk_message["content"], end="", flush=True)

    full_reply_content = "".join([m.get("content", "")
                                 for m in collected_messages])
    return messages + [
        {
            "role": "assistant",
            "content": full_reply_content,
        }
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float,
                        default=0.8,
                        help="temperature parameter between 0 and 2")
    args = parser.parse_args()

    messages = []
    while True:
        try:
            print("")
            text = input("User input: ")
            if not messages:
                sorted_articles = articles_relevance_order(text)
                messages = compose_prompt(sorted_articles, text)
            else:
                messages += [{"role": "user", "content": text}]
            print("Reply from GPT: ")
            messages = call_gpt(messages, temperature=args.temperature)
            save_json(messages, "chat_log", quiet=True)
        except KeyboardInterrupt:
            break
