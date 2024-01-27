import argparse
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
        try:
            with open(os.path.join(text_dir, file), "r", encoding="UTF-8") as f:
                text = f.read()
                texts.append(
                    (file.replace("-", " ").replace("_", " ").replace("#update", ""), text)
                )
        except UnicodeDecodeError:
            continue
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
            chunk_join = ". ".join(chunk) + "."
            chunks.append([chunk_join, len(tokenizer.encode(chunk_join))])
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
            many = split_into_many(row[1]["text"], max_tokens)
            for m in many:
                shortened.append(m)
        else:
            shortened.append([row[1]["text"][0], row[1]["n_tokens"]])

    df = pd.DataFrame(shortened, columns=["text", "n_tokens"])
    return df


def get_embedding(text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def embed(df, csv_path, model="text-embedding-3-small"):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    df["embeddings"] = df.text.apply(lambda x: get_embedding(x, client, model=model))
    df.to_csv(csv_path)


def main(knowledge_dir, embedding_csv, max_tokens, model):
    df = text2df(knowledge_dir)
    shorten_df = shorten(df, max_tokens)
    embed(shorten_df, embedding_csv, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings of knowledge")

    parser.add_argument("--knowledge_dir", type=str, help="Directory of the knowledge")

    parser.add_argument(
        "--embedding_csv", type=str, help="Absoluate pth to the embedding csv"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to pass the embedding engine",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding engine from OpenAI",
    )

    args = parser.parse_args()

    main(args.knowledge_dir, args.embedding_csv, args.max_tokens, args.model)
