open Argu

open Embeddings
open Prompt
open Chat

type Arguments =
    | Keyword of string
    | Question of string
    | Chat
    | ChatNoReference

    interface IArgParserTemplate with
        member s.Usage =
            match s with
            | Keyword _ -> "keyword used to search news article"
            | Question _ -> "question to ask to ChatGPT"
            | Chat -> "Chat over the reference news articles"
            | ChatNoReference -> "Chat without using reference news articles"

// dotnet run --keyword OpenAI
// dotnet run --question "What can ChatGPT do?"
// dotnet run --chat
[<EntryPoint>]
let main args =
    let parser = ArgumentParser.Create<Arguments>()
    let inputs = (parser.Parse args).GetAllResults()
    inputs |> List.map(fun x ->
        match x with
        | Keyword word ->
            let embeddedArtices = loadNews word
            saveEmbeddedArticles embeddedArtices
        | Question question ->
            let sortedArticles = articlesRelevanceOrder question
            let questionPrompt = composePrompt sortedArticles question
            saveJson questionPrompt "fsharp_prompt" false
        | Chat ->
            chat 0.8 false
        | ChatNoReference ->
            chat 0.8 true

    ) |> ignore
    0
