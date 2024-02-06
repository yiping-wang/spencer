from redis.commands.search.field import TextField, NumericField, VectorField
from redis.commands.search.query import Query

VECTOR_DIMENSION = 1536
SCHEMA = (
    NumericField("$.id", as_name="id"),
    TextField("$.fid", no_stem=True, as_name="fid"),
    TextField("$.file_name", no_stem=True, as_name="file_name"),
    TextField("$.file_path", no_stem=True, as_name="file_path"),
    TextField("$.last_modified_time", no_stem=True, as_name="last_modified_time"),
    NumericField("$.n_tokens", as_name="n_tokens"),
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