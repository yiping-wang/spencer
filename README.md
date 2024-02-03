# spencer
- based on https://platform.openai.com/docs/tutorials/web-qa-embeddings
- based on https://redis.io/docs/get-started/vector-database/

# redis
- https://hub.docker.com/r/redis/redis-stack-server/
- First time setup
    - `docker pull redis/redis-stack-server`
    - `docker run -d --name redis-stack -p 6379:6379  -v ./redis-data:/data redis/redis-stack-server:latest`
    - `docker stop IMAGE_ID`
- Next time setup
    - `docker run -it -p 6379:6379  -v ./redis-data:/data IMAGE_ID`
    - `docker stop IMAGE_ID`

# openai
- `export OPENAI_API_KEY=KEY`

# install
- naviage to this folder
- `pip install .`

# usage
```
import spencer


```