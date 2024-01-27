import pandas as pd
import tiktoken
import os
from openai import OpenAI


max_tokens = 500


def remove_newlines(serie):
    serie = serie.str.replace("\n", " ")
    serie = serie.str.replace("\\n", " ")
    serie = serie.str.replace("  ", " ")
    serie = serie.str.replace("  ", " ")
    return serie


def text2df(text_dir):
    texts = []

    for file in os.listdir(text_dir):
        with open(os.path.join(text_dir, file), "r", encoding="UTF-8") as f:
            text = f.read()
            texts.append(
                (file.replace("-", " ").replace("_", " ").replace("#update", ""), text)
            )
    df = pd.DataFrame(texts, columns=["fname", "text"])

    df["text"] = df.fname + ". " + remove_newlines(df.text)

    tokenizer = tiktoken.get_encoding("cl100k_base")

    df["n_tokens"] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    return df


def split_into_many(text, max_tokens=500):
    sentences = text.split(". ")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def shorten(df, max_tokens=500):
    shortened = []

    for row in df.iterrows():
        if row[1]["text"] is None:
            continue
        if row[1]["n_tokens"] > max_tokens:
            shortened += split_into_many(row[1]["text"])
        else:
            shortened.append(row[1]["text"])

    df = pd.DataFrame(shortened, columns=["text"])
    return df


def embed(df, csv_path, model="text-embedding-ada-002"):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    df["embeddings"] = df.text.apply(
        lambda x: client.embeddings.create(input=[x], model=model)["data"][0][
            "embedding"
        ]
    )
    df.to_csv(csv_path)
