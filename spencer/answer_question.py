import argparse
import pandas as pd
import os
from openai import OpenAI
import numpy as np


def read(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return f"An error occurred: {e}"


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def read_embedding_csv(embedding_csv_path):
    df = pd.read_csv(embedding_csv_path, index_col=0)
    df["embeddings"] = df["embeddings"].apply(eval).apply(np.array)
    return df


def create_context(client, question, df, max_len, model):
    q_embeddings = (
        client.embeddings.create(input=[question], model=model).data[0].embedding
    )

    df["distances"] = df.embeddings.apply(lambda x: cosine_similarity(x, q_embeddings))

    returns = []
    cur_len = 0

    for i, row in df.sort_values("distances", ascending=True).iterrows():
        cur_len += row["n_tokens"] + 4

        if cur_len > max_len:
            break

        returns.append(row["text"])

    return "\n\n###\n\n".join(returns)


def main(
    embedding_csv,
    system_instruct_path,
    model="gpt-3.5-turbo",
    question="I am looking for a credit card.",
    additional_context="",
    embed_model="text-embedding-3-small",
    max_len=1000,
    debug=True,
    max_tokens=500,
    stop_sequence=None,
):
    df = read_embedding_csv(embedding_csv)
    system_instruct = read(system_instruct_path)
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    context = create_context(
        client,
        question,
        df,
        max_len=max_len,
        model=embed_model,
    )
    if debug:
        print(system_instruct)
        print(
            f"Context: {context}\n\n---\n\n{additional_context}\n\nQuestion: {question}\nAnswer:"
        )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_instruct,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\n---\n\n{additional_context}\n\nQuestion: {question}\nAnswer:",
                },
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response
    except Exception as e:
        print(e)
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions with context")

    parser.add_argument(
        "--embedding_csv", type=str, help="Absoluate pth to the embedding csv"
    )

    parser.add_argument(
        "--system_instruct_path",
        type=str,
        default="system_instruct.txt",
        help="Absoluate pth to the system_instruct",
    )

    parser.add_argument(
        "--chat_model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model for chatting",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="Can you recommand a credit card for me? I travel a lot.",
        help="Question to ask",
    )

    parser.add_argument(
        "--embed_model",
        type=str,
        default="text-embedding-3-small",
        help="Model for embedding",
    )

    args = parser.parse_args()

    resp = main(
        args.embedding_csv,
        args.system_instruct_path,
        args.chat_model,
        args.question,
        args.additional_context,
        args.embed_model,
    )

    print(resp)
