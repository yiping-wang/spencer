import redis
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from openai import OpenAI
import tiktoken
from . import constants
import numpy as np
import pandas as pd
import argparse
import os


class Searcher:
    def __init__(
        self,
        redis_client,
        openai_client,
        embed_model,
        max_context_len,
        key_prefix="key",
        knn=20,
    ):
        self.r = redis_client
        self.o = openai_client
        self.em = embed_model
        self.mcl = max_context_len
        self.t = tiktoken.get_encoding("cl100k_base")
        self.kp = key_prefix
        self.knn = knn
        self.index_name = f"idx:{self.kp}_vss"

        try:
            self.r.ft(self.index_name).info()
        except redis.ResponseError:
            self.create_index()

    def create_index(self):
        schema = constants.SCHEMA
        definition = IndexDefinition(prefix=[f"{self.kp}:"], index_type=IndexType.JSON)
        self.r.ft(self.index_name).create_index(fields=schema, definition=definition)
        info = self.r.ft(self.index_name).info()
        num_docs = info["num_docs"]
        indexing_failures = info["hash_indexing_failures"]
        print(num_docs)
        print(indexing_failures)

    def create_query_table(self, query, embedded_query, extra_params={}):
        results_list = []

        result_docs = (
            self.r.ft(self.index_name)
            .search(
                Query(f"(*)=>[KNN {self.knn} @vector $query_vector AS vector_score]")
                .sort_by("vector_score")
                .return_fields(
                    "vector_score",
                    "id",
                    "fid",
                    "file_name",
                    "file_path",
                    "last_modified_time",
                    "n_tokens",
                    "description",
                )
                .dialect(2),
                {"query_vector": np.array(embedded_query, dtype=np.float32).tobytes()}
                | extra_params,
            )
            .docs
        )
        for doc in result_docs:
            vector_score = round(1 - float(doc.vector_score), 2)
            results_list.append(
                {
                    "query": query,
                    "score": vector_score,
                    "id": doc.id,
                    "fid": doc.fid,
                    "file_name": doc.file_name,
                    "file_path": doc.file_path,
                    "n_tokens": doc.n_tokens,
                    "last_modified_time": doc.last_modified_time,
                    "description": doc.description,
                }
            )

        queries_table = pd.DataFrame(results_list)
        queries_table.sort_values(by=["score"], ascending=[True], inplace=True)
        return queries_table

    def find(self, question):
        # embedded_question = np.random.randn(constants.VECTOR_DIMENSION)
        results = {}
        embedded_question = (
            self.o.embeddings.create(input=[question], model=self.em).data[0].embedding
        )
        df = self.create_query_table(question, embedded_question)
        cur_len = 0
        for _, row in df.sort_values("score", ascending=True).iterrows():
            # at least 1 context
            results[row["fid"]] = row["description"]
            cur_len += int(row["n_tokens"]) + 4
            if cur_len > self.mcl:
                break
        return results


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
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding engine from OpenAI",
    )
    parser.add_argument(
        "--max_context_len",
        type=int,
        default=500,
        help="Max length of the context",
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
    searcher = Searcher(
        r,
        o,
        args.embedding_model,
        args.max_context_len,
        args.key_prefix,
    )
    print(searcher.find(args.question))
