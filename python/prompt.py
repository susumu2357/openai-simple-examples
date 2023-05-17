import argparse
from operator import mul
from typing import Dict, List

from embeddings import (
    EmbeddedArticle,
    call_embedding,
    load_embedded_articles,
    save_json,
)

MAX_CHARS = 10_000  # about 2500 tokens in English

BASE = """You are a helpful assistant to answer to a question based on the news articles provided.
The question follows after <<question>>, where 'question' is a fixed value.
The reference news articles are provided with <<source_name>>, where 'source_name' is a title of the news article.
If you cannot answer the question based on the reference news articles, do not make up information;
suggest possible topics that can be answered based on the news articles provided.
""".replace("\n", " ")

SYSTEM_PROMPT = [
    {"role": "system", "content": BASE},
]


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


def compose_prompt(sorted_articles: List[EmbeddedArticle], question: str) -> List[Dict[str, str]]:
    num_chars = 0
    references = []

    for article in sorted_articles:
        num_chars += len(article.article.text)
        if num_chars > MAX_CHARS:
            break
        references.append(f"""<<{article.article.title.strip()}>>
                          {article.article.text.strip()}""")

    reference_text = "\n\n".join(references)
    prompt = SYSTEM_PROMPT + [
        {
            "role": "user",
            "content": reference_text + f"\n\n<<question>>\n{question}",
        }
    ]
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str,
                        help="question to ask to ChatGPT")
    args = parser.parse_args()

    sorted_articles = articles_relevance_order(args.question)
    question_prompt = compose_prompt(sorted_articles, args.question)
    save_json(question_prompt,
              f"prompt_for_Q_{args.question.replace(' ', '_')}")
