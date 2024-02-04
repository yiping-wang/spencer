# spencer
spencer is a simple yet get-it-done library that embed any knowledge (e.g., text files) to a searchable database, and allow q&a on the embedded knowledge. 

spencer uses redis as vector database, allowing ultra-fast vector similarity searches and retrieval operations. 

# redis
- https://hub.docker.com/r/redis/redis-stack-server/
- 1st time
    - `docker pull redis/redis-stack-server`
    - `docker run -d --name redis-stack -p 6379:6379  -v ./redis-data:/data redis/redis-stack-server:latest`
    - `docker stop IMAGE_ID`
- onwards
    - `docker run -it -p 6379:6379  -v ./redis-data:/data IMAGE_ID`
    - `docker stop IMAGE_ID`

# openai
- `export OPENAI_API_KEY=KEY`

# install
- naviage to this folder
- `pip install .`

# usage
## embedding
prepare a folder that contains all the knowledge. this folder can be nested. 

```
from spencer import Embedder
r = redis.Redis(host='locahost', port=6379, decode_responses=True)
o = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embedder = Embedder(
    r, o, "text-embedding-3-small", knowledge_dir, max_tokens=2000
)
success = embedder()
print(success)
```

verify whether the knowledge has been embedded or not via `redis-cli`

the knowledge keys are in format `knowledge:doc_id:chunk_id`. 

## spencer
spencer automatically embed the question, and find the closest `k` knowledge vector and its text. 

spencer uses these `k` texts as context. further, spencer allows `adhoc_context`, meaning you can 

pass context other than the knolwedge we can just embedded. 

```
r = redis.Redis(host='locahost', port=6379, decode_responses=True)
o = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
spencer_client = Spencer(
    r,
    o,
    system_instruction,
    chat_model,
    embedding_model,
    max_context_len,
    max_tokens,
)

question_1 = '...'
resp = spencer_client.answer(question_1, adhoc_context="...")

question_2 = '...'
resp = spencer_client.answer(question_2, adhoc_context="...")
```

# reference
- based on https://platform.openai.com/docs/tutorials/web-qa-embeddings
- based on https://redis.io/docs/get-started/vector-database/


