import redis
from redis.commands.search.field import TextField, NumericField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import NumericFilter, Query
import pandas as pd
import numpy as np

VECTOR_DIMENSION = 1536


def create_query_table(query, queries, encoded_queries, extra_params={}):
    results_list = []
    for i, encoded_query in enumerate(encoded_queries):
        result_docs = (
            client.ft("idx:knowledge_vss")
            .search(
                query,
                {"query_vector": np.array(encoded_query, dtype=np.float32).tobytes()}
                | extra_params,
            )
            .docs
        )
        for doc in result_docs:
            print(doc)
            vector_score = round(1 - float(doc.vector_score), 2)
            results_list.append(
                {
                    "query": queries[i],
                    "score": vector_score,
                    "file_name": doc.file_name,
                    "file_path": doc.file_path,
                    "description": doc.description,
                }
            )

    # Optional: convert the table to Markdown using Pandas
    queries_table = pd.DataFrame(results_list)
    queries_table.sort_values(
        by=["query", "score"], ascending=[True, False], inplace=True
    )
    queries_table["query"] = queries_table.groupby("query")["query"].transform(
        lambda x: [x.iloc[0]] + [""] * (len(x) - 1)
    )
    queries_table["description"] = queries_table["description"].apply(
        lambda x: (x[:497] + "...") if len(x) > 500 else x
    )
    return queries_table


client = redis.Redis(host="localhost", port=6379, decode_responses=True)
schema = (
    TextField("$.file_name", no_stem=True, as_name="file_name"),
    TextField("$.file_path", no_stem=True, as_name="file_path"),
    TextField("$.last_modified_time", no_stem=True, as_name="last_modified_time"),
    NumericField("$.n_tokens", as_name="n_tokens"),
    NumericField("$.chunk_id", as_name="chunk_id"),
    TextField("$.description", as_name="description"),
    VectorField(
        "$.description_embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIMENSION,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
)
definition = IndexDefinition(prefix=["knowledge:"], index_type=IndexType.JSON)
res = client.ft("idx:knowledge_vss").create_index(fields=schema, definition=definition)

info = client.ft("idx:knowledge_vss").info()
num_docs = info["num_docs"]
indexing_failures = info["hash_indexing_failures"]
print(num_docs)
print(indexing_failures)

queries = ["a", "b"]
encoded_queries = [np.random.randn(1536), np.random.randn(1536)]

query = (
    Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "file_name", "file_path", "description")
    .dialect(2)
)

query = (
    Query("(*)=>[KNN 3 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields("vector_score", "id", "file_name", "file_path", "description")
    .dialect(2)
)

t = create_query_table(query, queries, encoded_queries)
print(t)
