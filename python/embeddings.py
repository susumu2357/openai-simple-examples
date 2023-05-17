"""
Fetch news articles that contain the keyword,
convert the title to a vector using the OpenAI Embedding API,
save the results in ../data/news_with_embeddings.json.

Example usage:
    python3 embeddings.py --keyword Microsoft
"""

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Union

import requests

NUM_NEWS = 100
START_DATETIME = "2023-05-01%2000:00:00"


@dataclass
class Embedding:
    vector: List[float]
    total_tokens: int


@dataclass
class Article:
    title: str
    text: str
    publish_date: str


@dataclass
class EmbeddedArticle:
    article: Article
    embedding: Embedding


def call_embedding(text: str) -> Embedding:
    url = "https://api.openai.com/v1/embeddings"
    key = os.environ["OPENAI_API_KEY"]
    headers = {"Authorization": f"Bearer {key}"}
    data = {
        "input": text,
        "model": "text-embedding-ada-002",
    }
    res = requests.post(url, headers=headers, json=data).json()

    vec: List[float] = res["data"][0]["embedding"]
    total_tokens: int = res["usage"]["total_tokens"]
    return Embedding(
        vector=vec,
        total_tokens=total_tokens,
    )


def save_json(data: Union[Dict, List[Dict]], filename: str) -> None:
    with open(f"../data/{filename}.json", "w") as file:
        json.dump(data, file, indent=4)
    print(f"Saved {filename}.json!")
    return


def fetch_news(keyword: str) -> List[Article]:
    key = os.environ["NEWS_API_KEY"]
    url = f"https://api.worldnewsapi.com/search-news?text={keyword}&language=en\
        &number={NUM_NEWS}&earliest-publish-date={START_DATETIME}&api-key={key}"
    res = requests.get(url).json()

    save_json(res, "news_response")

    articles = [
        Article(title=article["title"], text=article["text"],
                publish_date=article["publish_date"])
        for article in res["news"]]
    return articles


def load_news(keyword: str) -> List[EmbeddedArticle]:
    articles = fetch_news(keyword)
    title_embeddings = [call_embedding(article.title) for article in articles]
    return [EmbeddedArticle(article=article, embedding=embedding)
            for article, embedding in zip(articles, title_embeddings)]


def save_embedded_articles(embedded_articles: List[EmbeddedArticle]) -> None:
    embedded_articles_dict = [dataclasses.asdict(
        embedded_article) for embedded_article in embedded_articles]
    save_json(embedded_articles_dict, "news_with_embeddings")
    return


def load_embedded_articles(data_path: str) -> List[EmbeddedArticle]:
    with open(data_path, "r") as file:
        embedded_articles_dict = json.load(file)

    embedded_articles = [
        EmbeddedArticle(
            article=Article(
                title=elm["article"]["title"],
                text=elm["article"]["text"],
                publish_date=elm["article"]["publish_date"]
            ),
            embedding=Embedding(
                vector=elm["embedding"]["vector"],
                total_tokens=elm["embedding"]["total_tokens"]
            )
        )
        for elm in embedded_articles_dict
    ]
    return embedded_articles


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str,
                        help="keyword used to search news article")
    args = parser.parse_args()

    embedded_articles = load_news(args.keyword)
    save_embedded_articles(embedded_articles)

    # print(embedded_articles)
