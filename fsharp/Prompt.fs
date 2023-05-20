module Prompt

open System

open Embeddings

let SYSTEM = "system"
let USER = "user"
let ASSISTANT = "assistant"

type Message ={
    role: string
    content: string
}

let MAX_CHARS = 10000

let SYSTEM_DESCRIPTION = "You are a helpful assistant to answer to a question based on the news articles provided. \
The question follows after <<question>>, where 'question' is a fixed value. \
One or more news articles are provided with <<source_name>>, where 'source_name' is a title of the news article. \
You must cite the title of the news article you are referring to at the end of your answer. \
If you cannot answer the question based on the reference news articles, do not make up information; \
suggest possible topics that can be answered based on the news articles provided.
"

let articlesRelevanceOrder (question: string) =
    let embeddedArticles = loadEmbeddedArticles "fsharp_news_with_embeddings.json"
    let qEmb = callEmbedding question
    let similarityScores =
        List.map (fun ref ->
            List.sum (List.map2 (fun x y -> x*y) qEmb.vector ref.embedding.vector)
            ) embeddedArticles
    List.sortByDescending fst (List.zip similarityScores embeddedArticles)
    |> List.map snd

let composePrompt (sortedArticles: EmbeddedArticle list) (question: string) =
    let mutable numChars = 0
    let mutable references = []
    let mutable Break = false

    for article in sortedArticles do
        numChars <- numChars + article.article.text.Length
        if Break then
            0 |> ignore
        elif numChars > MAX_CHARS then
            references <- references @ [sprintf "<<%s>>\n%s" article.article.title (article.article.text.Substring(0, article.article.text.Length - (numChars - MAX_CHARS)))]
            Break <- true
        else
            references <- references @ [sprintf "<<%s>>\n%s" article.article.title article.article.text]

    let referenceText = String.Join("\n\n", references)
    [
        { role = SYSTEM; content = SYSTEM_DESCRIPTION };
        { role = USER; content = referenceText + "\n\n<<question>>\n" + question }
    ]