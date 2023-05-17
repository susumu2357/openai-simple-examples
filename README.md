# openai-simple-examples

## 1. Prepare reference
- Fetch recent news articles using a public API, such as [World News API](https://worldnewsapi.com/).
- Convert the title text to an embedding vector using [the OpenAI embedding API](https://platform.openai.com/docs/api-reference/embeddings).

An example code is [python/embeddings.py](python/embeddings.py).  
The code uses the `requests` to call the OpenAI API instead of the `openai` to make it easier to translate from Python to other languages such as C# that do not have an official OpenAI library.
Note that, there are [community-maintained libraries](https://platform.openai.com/docs/libraries/community-libraries) for most of major languages.

## 2. Compose prompt
- Convert the question to an embedding vector using the OpenAI embedding API.
- Compute the inner products of the question vector and the title vectors. These values represent the similarities between the question and the titles.
- Sort the articles by the similarity.
- Append the news text until the length reaches the threshold.
- Append the question at the last.

An example code is [python/prompt.py](python/prompt.py).  
You can play with `SYSTEM_DESCRIPTION` to see how the response changes. According to [the course by Andrew Ng](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/), there are two general principles of prompt engineering.  
1. Write clear and specific instructions
2. Give the model time to “think”

## 3. Chat over the reference texts
- Receive the input text and compose the prompt.
- Call [the OpenAI chat API](https://platform.openai.com/docs/api-reference/chat) and print out the stream responses.
- Continue the chat until the code is terminated.

An example code is [python/chat.py](python/chat.py).  
Again, we use the `requests` to call the OpenAI API instead of the `openai`. The `call_gpt` function may need to be modified when translated into other languages.