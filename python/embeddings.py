import json
import os
from dataclasses import dataclass
from typing import List

import requests

PRICE_ADA = 0.0004 / 1000


@dataclass
class Embedding:
    embedding: List[float]
    total_tokens: int


@dataclass
class Article:
    title: str
    publishedAt: str
    description: str


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

    vec: List[float] = res.data.embedding
    total_tokens: int = res.usage.total_tokens
    return Embedding(
        embedding=vec,
        total_tokens=total_tokens,
    )


def fetch_news(keyword: str) -> List[Article]:
    key = os.environ["NEWS_API_KEY"]
    url = f"https://newsapi.org/v2/everything?q={keyword}&language=en&apiKey={key}"
    res = requests.get(url).json()

    articles: List[Article] = [
        Article(title=article.title, publishedAt=article.publishedAt,
                description=article.description)
        for article in res.articles]
    return articles


def load_news(keyword: str) -> List[EmbeddedArticle]:
    articles = fetch_news(keyword)
    title_embeddings = [call_embedding(article.title) for article in articles]
    return [EmbeddedArticle(article=article, embedding=embedding)
            for article, embedding in zip(articles, title_embeddings)]


def save_embedded_articles(embedded_articles: List[EmbeddedArticle]) -> None:
    embedded_articles_dict = [dataclass.asdict(
        embedded_article) for embedded_article in embedded_articles]
    with open("../data/embedded_articles.json", "w") as file:
        json.dump(embedded_articles_dict, file)
    return


def load_embedded_articles(data_path: str) -> List[EmbeddedArticle]:
    with open(data_path, "r") as file:
        embedded_articles_dict = json.load(file)

    embedded_articles = [
        EmbeddedArticle(
            article=Article(
                title=elm.article.title,
                publishedAt=elm.article.publishedAt,
                description=elm.article.description
            ),
            embedding=Embedding(
                embedding=elm.embedding.embedding,
                total_tokens=elm.embedding.total_tokens
            )
        )
        for elm in embedded_articles_dict
    ]
    return embedded_articles
