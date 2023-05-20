module Embeddings

open System
open System.IO
open System.Net
open System.Net.Http
open System.Text.Json
open System.Collections.Generic

open Argu

type Embedding = {
    vector: float list
    total_tokens: int
}

type Article = {
    title: string
    text: string
    publish_date: string
}

type EmbeddedArticle = {
    article: Article
    embedding: Embedding
}

type OpenAIEmbeddingData = {
    input: string
    model: string
}

let NUM_NEWS = 100
let START_DATETIME = "2023-05-01%2000:00:00"

let callEmbedding (text: string)  =
    let url = "https://api.openai.com/v1/embeddings"
    let key = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
    let headers = new WebHeaderCollection()
    headers.Add("Authorization", "Bearer " + key)
    let req = WebRequest.Create(url)
    req.Method <- "POST"
    req.Headers <- headers
    req.ContentType <- "application/json"
    use streamWriter = new StreamWriter(req.GetRequestStream())
    let data = JsonSerializer.Serialize(
        {
            input=text
            model="text-embedding-ada-002"
        })
    streamWriter.Write(data)
    streamWriter.Flush()
    let res = req.GetResponse()
    use streamReader = new StreamReader(res.GetResponseStream())
    let resJson = streamReader.ReadToEnd()
    let resObj = JsonSerializer.Deserialize<Dictionary<string, obj>>(resJson)
    let vec = resObj.["data"] :?> JsonElement
            |> fun x -> x.EnumerateArray()
            |> Seq.map (fun x -> x.GetProperty("embedding"))
            |> Seq.head
            |> fun x -> x.EnumerateArray()
            |> Seq.map(fun x -> x.GetDouble())
            |> Seq.toList
    let total_tokens =
        resObj.["usage"] :?> JsonElement
        |> fun x -> x.GetProperty("total_tokens").GetInt32()
    { vector = vec; total_tokens = total_tokens }

let saveJson (data: obj) (filename: string) (quiet: bool) : unit =
    let path = "../data/" + filename + ".json"
    use streamWriter = new FileStream(path, FileMode.Create)
    let options = JsonSerializerOptions()
    options.WriteIndented <- true
    JsonSerializer.Serialize(streamWriter, data, options)
    if not quiet then printfn "Saved %s!" filename

let fetchNews (keyword: string) : Article list =
    let key = Environment.GetEnvironmentVariable("NEWS_API_KEY")
    let url = sprintf "https://api.worldnewsapi.com/search-news?text=%s&language=en&number=%d&earliest-publish-date=%s&api-key=%s" keyword NUM_NEWS START_DATETIME key
    let client = new HttpClient()
    let res =
        client.GetAsync(url)
        |> Async.AwaitTask
        |> Async.RunSynchronously
    let resObj =
        res.Content.ReadAsStringAsync()
        |> Async.AwaitTask
        |> Async.RunSynchronously
        |> JsonSerializer.Deserialize<Dictionary<string, obj>>
    let articles = resObj.["news"] :?> JsonElement
                    |> fun x -> x.EnumerateArray()
                    |> Seq.map (fun x -> {
                        title = x.GetProperty("title").GetString()
                        text = x.GetProperty("text").GetString()
                        publish_date = x.GetProperty("publish_date").GetString()
                    })
                    |> Seq.toList
    saveJson articles "fsharp_articles" false
    articles

let loadNews (keyword: string) : EmbeddedArticle list =
    let articles = fetchNews keyword
    let titleEmbeddings =
        articles
        |> List.map (fun article -> callEmbedding article.title)
    List.map2 (fun article embedding -> {
        article = article
        embedding = embedding
    }) articles titleEmbeddings

let saveEmbeddedArticles (embedded_articles: EmbeddedArticle list) : unit =
    let embeddedArticlesDict = embedded_articles |> List.map (fun embeddedArticle -> {
        article = embeddedArticle.article
        embedding = embeddedArticle.embedding
    })
    saveJson embeddedArticlesDict "fsharp_news_with_embeddings" false

let loadEmbeddedArticles (data_path: string) : EmbeddedArticle list =
    let path = "../data/" + data_path
    use streamReader = new StreamReader(path)
    let resJson = streamReader.ReadToEnd()
    let resObj = JsonSerializer.Deserialize<EmbeddedArticle list>(resJson)
    resObj
