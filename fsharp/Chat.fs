module Chat

open System
open System.IO
open System.Net
open System.Text.Json
open System.Collections.Generic

open Embeddings
open Prompt

type OpenAIChatCompletionData = {
    messages: Message list
    temperature: float
    model: string
    stream: bool
}

let callGpt (messages: Message list) (temperature: float) =
    let url = "https://api.openai.com/v1/chat/completions"
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
            messages = messages
            temperature = temperature
            model = "gpt-3.5-turbo"
            stream = false
            }
            )
    streamWriter.Write(data)
    streamWriter.Flush()
    let res = req.GetResponse()
    use streamReader = new StreamReader(res.GetResponseStream())
    let resJson = streamReader.ReadToEnd()
    let response = JsonSerializer.Deserialize<Dictionary<string, obj>>(resJson)
    let choices = response.Item("choices") :?> JsonElement
    let content =
        choices
        |> fun x -> x.EnumerateArray()
        |> Seq.head
        |> fun x -> x.GetProperty("message")
        |> fun x -> x.GetProperty("content").GetString()
    printf "%s" content
    messages @ [{ role = ASSISTANT; content = content }]

let chat temperature noReference =
    let mutable messages = List.empty
    let mutable text = ""
    while true do
        try
            printf "User input: "
            text <- Console.ReadLine()

            if List.length messages = 0 then
                if noReference then
                    messages <- messages @ [{ role = USER; content = text }]
                else
                    let sortedArticles = articlesRelevanceOrder text
                    messages <- composePrompt sortedArticles text
            else
                messages <- messages @ [{ role = USER; content = text }]

            printf "Reply from GPT: "
            messages <- callGpt messages temperature
            saveJson messages "fsharp_chat_log" true
            printfn ""
        with
        | :? IOException as _ -> ()
        | _ -> ()
