from .searcher import Searcher
import uuid
import redis
from openai import OpenAI
import argparse
import os


def read(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return f"An error occurred: {e}"


class Spencer:
    def __init__(
        self,
        redis_client,
        openai_client,
        system_instruction,
        chat_model,
        embedding_model,
        max_context_len,
        max_tokens,
    ):
        self.r = redis_client
        self.o = openai_client
        self.cm = chat_model
        self.em = embedding_model
        self.mcl = max_context_len
        self.mt = max_tokens
        self.searcher = Searcher(self.r, self.o, self.em, self.mcl)
        self.conversion_history = [
            {
                "role": "system",
                "content": system_instruction,
            }
        ]

    def answer(self, question, additional_context=""):
        db_context = self.searcher.find(question)
        if additional_context != "":
            prompt = f"Context: {db_context}\n\n---\n\n{additional_context}\n\nQuestion: {question}"
        else:
            prompt = f"Context: {db_context}\n\n---\n\nQuestion: {question}"
        self.conversion_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        response = self.o.chat.completions.create(
            model=self.cm,
            messages=self.conversion_history,
            temperature=0,
            max_tokens=self.mt,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
        )
        self.conversion_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings of knowledge")
    parser.add_argument("--redis_host", type=str, default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument(
        "--question",
        type=str,
        default="Can you recommand a credit card for me? I travel a lot.",
        help="Question to ask",
    )
    parser.add_argument(
        "--system_instruction_path",
        type=str,
        default="./system_instruct.txt",
        help="Question to ask",
    )
    parser.add_argument(
        "--chat_model",
        type=str,
        # default="gpt-3.5-turbo", # gpt-4-turbo-preview
        default="gpt-4-turbo-preview",
        help="Embedding engine from OpenAI",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding engine from OpenAI",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=5000,
        help="Max length of the context",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2500,
        help="Max length of the context",
    )
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    o = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    spencer = Spencer(
        r,
        o,
        read(args.system_instruction_path),
        args.chat_model,
        args.embedding_model,
        args.max_context_len,
        args.max_tokens,
    )
    print(spencer.answer(args.question))
    print(spencer.conversion_history)
