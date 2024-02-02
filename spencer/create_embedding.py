import argparse
import pandas as pd
import tiktoken
import os
import base64
import redis
from openai import OpenAI
import datetime

EMBEDDING_DIM = 1536


def encode(file_path):
    with open(file_path, "rb") as file:
        binary_content = file.read()
        return base64.b64encode(binary_content)


def remove_newlines(s):
    s = s.replace("\n", " ")
    s = s.replace("\\n", " ")
    s = s.replace("  ", " ")
    s = s.replace("  ", " ")
    return s


def find_knowledge_to_embed(redis_client, knowledge_dir):
    knowledge_loc = {}
    for file in os.listdir(knowledge_dir):
        file_path = os.path.join(knowledge_dir, file)
        encoding = encode(file_path)
        id = file_path + ":hash"
        prev_encoding = redis_client.get(id)
        if prev_encoding == encoding:
            continue
        redis_client.set(id, encoding)
        knowledge_loc[file] = file_path
    return knowledge_loc


def break_knowledge(
    file_name, file_path, knowledge_id, model, openai_client, redis_pipeline, max_tokens
):
    file_name = file_name.replace("-", " ").replace("_", " ").replace("#update", "")
    print(file_path)
    with open(file_path, "r", encoding="UTF-8") as f:
        text = f.read()
    text = remove_newlines(text)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    n_tokens = len(tokenizer.encode(text))

    if n_tokens > max_tokens:
        chunks, knowledge_id = split_into_many(
            knowledge_id, text, model, openai_client, max_tokens
        )
        for i, chunk in chunks.items():
            chunk["file_name"] = file_name
            chunk["file_path"] = file_path
            chunk["last_modified_time"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).strftime("%Y-%m-%d %H:%M:%S")
            redis_key = f"knowledge:{i:05}"
            redis_pipeline.json().set(redis_key, "$", chunk)
    else:
        chunk = {
            "file_name": file_name,
            "file_path": file_path,
            "last_modified_time": datetime.datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).strftime("%Y-%m-%d %H:%M:%S"),
            "n_tokens": len(tokenizer.encode(text)),
            "chunk_id": 0,
            "description": text,
            "description_embedding": openai_client.embeddings.create(
                input=[text.replace("\n", " ")], model=model
            )
            .data[0]
            .embedding,
        }
        redis_key = f"knowledge:{knowledge_id:05}"
        redis_pipeline.json().set(redis_key, "$", chunk)
        knowledge_id += 1

    return knowledge_id


def split_into_many(knowledge_id, text, model, openai_client, max_tokens):
    sentences = text.split(". ")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = {}
    chunk_id = 0
    tokens_so_far = 0
    chunk = []

    for sentence, n_token in zip(sentences, n_tokens):
        if tokens_so_far + n_token > max_tokens:
            cur_sentence = ". ".join(chunk) + "."
            chunk = []
            tokens_so_far = 0
            chunks[knowledge_id] = {
                "n_tokens": len(tokenizer.encode(cur_sentence)),
                "chunk_id": chunk_id,
                "description": cur_sentence,
                "description_embedding": openai_client.embeddings.create(
                    input=[sentence.replace("\n", " ")], model=model
                )
                .data[0]
                .embedding,
            }
            chunk_id += 1
            knowledge_id += 1

        # still want that knowledge even if it is too big
        if n_token > max_tokens:
            chunks[knowledge_id] = {
                "n_tokens": len(tokenizer.encode(sentence)),
                "chunk_id": 0,
                "description": sentence,
                "description_embedding": openai_client.embeddings.create(
                    input=[sentence.replace("\n", " ")], model=model
                )
                .data[0]
                .embedding,
            }
            knowledge_id += 1

        chunk.append(sentence)
        tokens_so_far += n_token + 1

    return chunks, knowledge_id


def main(
    redis_host,
    redis_port,
    knowledge_dir,
    max_tokens,
    model,
):
    knowledge_id = 0
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    c = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    pipeline = r.pipeline()
    knowledge_loc = find_knowledge_to_embed(r, knowledge_dir)
    for file_name, file_path in knowledge_loc.items():
        knowledge_id = break_knowledge(
            file_name, file_path, knowledge_id, model, c, pipeline, max_tokens
        )
    pipeline.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings of knowledge")

    parser.add_argument("--redis_host", type=str, default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)

    parser.add_argument(
        "--knowledge_dir",
        type=str,
        help="Directory of the knowledge",
        default="/Users/yiping/Projects/pointer_knowledge",
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

    main(
        args.redis_host,
        args.redis_port,
        args.knowledge_dir,
        args.max_tokens,
        args.model,
    )
