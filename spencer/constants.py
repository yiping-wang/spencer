from redis.commands.search.field import TextField, NumericField, VectorField
from redis.commands.search.query import Query

VECTOR_DIMENSION = 1536
SCHEMA = (
    NumericField("$.id", as_name="id"),
    TextField("$.file_name", no_stem=True, as_name="file_name"),
    TextField("$.file_path", no_stem=True, as_name="file_path"),
    TextField("$.last_modified_time", no_stem=True, as_name="last_modified_time"),
    NumericField("$.n_tokens", as_name="n_tokens"),
    NumericField("$.chunk_id", as_name="chunk_id"),
    TextField("$.description", as_name="description"),
    VectorField(
        "$.embedding",
        "FLAT",
        {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIMENSION,
            "DISTANCE_METRIC": "COSINE",
        },
        as_name="vector",
    ),
)
QUERY = (
    Query("(*)=>[KNN 20 @vector $query_vector AS vector_score]")
    .sort_by("vector_score")
    .return_fields(
        "vector_score",
        "id",
        "file_name",
        "file_path",
        "last_modified_time",
        "n_tokens",
        "description",
    )
    .dialect(2)
)
INDEX_NAME = "idx:knowledge_vss"
