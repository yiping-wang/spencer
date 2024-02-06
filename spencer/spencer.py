from .searcher import Searcher
import uuid
import redis
from openai import OpenAI
import argparse
import os


def read_f(fp):
    try:
        with open(fp, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"{fp} was not found."
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
        key_prefix,
        knn,
    ):
        self.r = redis_client
        self.o = openai_client
        self.cm = chat_model
        self.em = embedding_model
        self.mcl = max_context_len
        self.mt = max_tokens
        self.kp = key_prefix
        self.searcher = Searcher(self.r, self.o, self.em, self.mcl, self.kp, knn)
        self.conversion_history = [
            {
                "role": "system",
                "content": system_instruction,
            }
        ]
        self.id = str(uuid.uuid4())

        # knowledge id in the conversion history
        self.kid_in_ch = set()

    def answer(self, question, adhoc_context=""):
        results = self.searcher.find(question)
        context_list = []
        for k, v in results.items():
            if k not in self.kid_in_ch:
                context_list.append(v)
                self.kid_in_ch.add(k)

        context = "\n\n###\n\n".join(context_list)
        if adhoc_context != "":
            prompt = (
                f"Context: {context}\n\n---\n\n{adhoc_context}\n\nQuestion: {question}"
            )
        else:
            prompt = f"Context: {context}\n\n---\n\nQuestion: {question}"
        self.conversion_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        response = (
            self.o.chat.completions.create(
                model=self.cm,
                messages=self.conversion_history,
                temperature=0,
                max_tokens=self.mt,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            .choices[0]
            .message.content
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
        help="Path to a system instruction file",
    )
    parser.add_argument(
        "--chat_model",
        type=str,
        default="gpt-3.5-turbo",
        help="Chat engine from OpenAI",
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
        help="Max tokens",
    )
    parser.add_argument(
        "--key_prefix",
        type=str,
        default="key",
        help="Redis key prefix",
    )
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    o = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    spencer = Spencer(
        r,
        o,
        read_f(args.system_instruction_path),
        args.chat_model,
        args.embedding_model,
        args.max_context_len,
        args.max_tokens,
        args.key_prefix,
    )
    print(spencer.answer(args.question))
    print(spencer.conversion_history)
