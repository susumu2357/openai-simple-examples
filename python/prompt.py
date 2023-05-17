"""
Compute the similarity between the question and reference news titles,
sort the articles by the similarity,
append the news content until the length reaches the MAX_CHARS,
append the question at the last,
save the prompt as a JSON file.

Example usage:
    python3 prompt.py --question "Is AI dangerous?"
"""

import argparse
from dataclasses import dataclass
from enum import Enum
from operator import mul
from typing import Dict, List

from embeddings import (
    EmbeddedArticle,
    call_embedding,
    load_embedded_articles,
    save_json,
)


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class Messages:
    messages: List[Message]

    def to_dict(self) -> List[Dict[str, str]]:
        return [
            {"role": message.role.value, "content": message.content}
            for message in self.messages
        ]

    def append(self, other_message: Message):
        return Messages(messages=self.messages + [other_message])


MAX_CHARS = 10_000  # about 2500 tokens in English

SYSTEM_DESCRIPTION = """You are a helpful assistant to answer to a question based on the news articles provided.
The question follows after <<question>>, where 'question' is a fixed value.
One or more news articles are provided with <<source_name>>, where 'source_name' is a title of the news article.
You must cite the title of the news article you are referring to at the end of your answer.
If you cannot answer the question based on the reference news articles, do not make up information;
suggest possible topics that can be answered based on the news articles provided.
""".replace("\n", " ")


embedded_articles = load_embedded_articles(
    "../data/news_with_embeddings.json")


def articles_relevance_order(question: str) -> List[EmbeddedArticle]:
    q_emb = call_embedding(question)
    similarity_scores = [
        sum(map(mul, q_emb.vector, ref.embedding.vector))
        for ref in embedded_articles
    ]
    sorted_articles = [
        ref for _, ref in sorted(
            zip(similarity_scores, embedded_articles),
            key=lambda pair: pair[0], reverse=True
        )
    ]
    return sorted_articles


def compose_prompt(sorted_articles: List[EmbeddedArticle], question: str) -> Messages:
    num_chars = 0
    references = []

    for article in sorted_articles:
        num_chars += len(article.article.text)
        if num_chars > MAX_CHARS:
            references.append(f"""<<{article.article.title}>>
                            {article.article.text[:-(num_chars - MAX_CHARS)]}""")
            break
        references.append(f"""<<{article.article.title}>>
                          {article.article.text}""")

    reference_text = "\n\n".join(references)
    messages = Messages([
        Message(role=Role.SYSTEM, content=SYSTEM_DESCRIPTION),
        Message(role=Role.USER,
                content=reference_text + f"\n\n<<question>>\n{question}"),
    ])
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str,
                        help="question to ask to ChatGPT")
    args = parser.parse_args()

    sorted_articles = articles_relevance_order(args.question)
    question_prompt = compose_prompt(sorted_articles, args.question)
    save_json(question_prompt.to_dict(),
              f"prompt_for_Q_{args.question.replace(' ', '_')}")
