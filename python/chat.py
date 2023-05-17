"""
Chat over the reference news articles.

Example usage:
    python3 chat.py
"""

import argparse
import json
import os

import requests
from embeddings import save_json
from prompt import Message, Messages, Role, articles_relevance_order, compose_prompt


def call_gpt(messages: Messages, temperature: float) -> Messages:
    url = "https://api.openai.com/v1/chat/completions"
    key = os.environ["OPENAI_API_KEY"]
    headers = {"Authorization": f"Bearer {key}"}
    data = {
        "messages": messages.to_dict(),
        "temperature": temperature,
        "model": "gpt-3.5-turbo",
        "stream": True
    }
    res = requests.post(url, headers=headers, json=data, stream=True)

    # Reference https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
    collected_messages = []
    for line in res.iter_lines():
        # line is a bytes object
        # b'data: {
        # "id":"chatcmpl-7HD38WE6otovhZ21h0A4hyaArrAO1",
        # "object":"chat.completion.chunk","created":1684336094,
        # "model":"gpt-3.5-turbo-0301",
        # "choices":[
        # {"delta":{"role":"assistant"},"index":0,"finish_reason":null}
        # ]}'
        try:
            chunk = json.loads(line.decode("utf-8").replace("data: ", ""))
            chunk_message = chunk["choices"][0]["delta"]
            if "content" in chunk_message.keys():
                collected_messages.append(chunk_message)
                print(chunk_message["content"], end="", flush=True)
        except Exception:
            continue

    full_content = "".join([m.get("content", "")
                            for m in collected_messages])
    return messages.append(
        Message(role=Role.ASSISTANT, content=full_content)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float,
                        default=0.8,
                        help="temperature parameter between 0 and 2")
    parser.add_argument("--no_reference", action="store_true",
                        help="do not use reference texts")
    args = parser.parse_args()

    messages = Messages(messages=[])
    while True:
        try:
            text = input("User input: ")
            if not messages.messages:
                if args.no_reference:
                    messages = messages.append(
                        Message(role=Role.USER, content=text))
                else:
                    sorted_articles = articles_relevance_order(text)
                    messages = compose_prompt(sorted_articles, text)
            else:
                messages = messages.append(
                    Message(role=Role.USER, content=text)
                )
            print("Reply from GPT: ")
            messages = call_gpt(messages, temperature=args.temperature)
            save_json(messages.to_dict(), "chat_log", quiet=True)
            print("")
        except KeyboardInterrupt:
            break
