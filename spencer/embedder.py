import argparse
import tiktoken
import os
import redis
from openai import OpenAI
import datetime
import hashlib
import re
import uuid
import logging

EMBEDDING_DIM = 1536
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def hash_file_sha256(fp):
    hash_sha256 = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(16384), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def remove_newlines(s):
    s = re.sub(r"(\\n|\n)+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def find_files(dir):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(dir):
        if ".git" in dirnames:
            dirnames.remove(".git")
        for filename in filenames:
            if filename == ".DS_Store":
                continue
            file_path = os.path.join(dirpath, filename)
            all_files.append(file_path)
    return all_files


def get_file_last_modified_time(fp):
    return datetime.datetime.fromtimestamp(os.path.getmtime(fp)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def get_first_uuid():
    return str(uuid.uuid4())[:4]


class Embedder:
    def __init__(
        self,
        redis_client,
        openai_client,
        embedding_model,
        knowledge_dir,
        max_tokens,
        key_prefix="key",
    ):
        self.r = redis_client
        self.rp = self.r.pipeline()
        self.o = openai_client
        self.em = embedding_model
        self.k_dir = knowledge_dir
        self.mt = max_tokens
        self.kp = key_prefix
        self.fid = {}
        self.t = tiktoken.get_encoding("cl100k_base")
        self.id = 0

    def get_file_metadata(self, fid, fn, fp):
        return {
            "fid": fid,
            "file_name": fn,
            "file_path": fp,
            "last_modified_time": get_file_last_modified_time(fp),
        }

    def get_file_key(self, fp):
        return "file:" + fp + ":checksum"

    def remove_file_knowledge(self, id):
        pattern = f"{self.kp}:{id}:*"
        for key in self.r.scan_iter(match=pattern):
            self.r.delete(key)

    def find(self):
        """Find the files to embed."""
        loc = {}
        for fp in find_files(self.k_dir):
            fn = os.path.basename(fp)
            checksum = hash_file_sha256(fp)
            checksum_key = self.get_file_key(fp)
            logging.info(f"checking {fp}")
            prev_checksum = self.r.get(checksum_key)
            if prev_checksum == checksum:
                logging.info(f"{fp} hasn't changed, and won't be embedded")
                continue
            self.r.set(checksum_key, checksum)
            logging.info(f"{fp} is changed or not embedded, and will be embedded")
            loc[fn] = fp
            if prev_checksum:
                self.remove_file_knowledge(prev_checksum[:4])

        return loc

    def split(self, text):

        # create metadata for each chunk
        def _info(self, s):
            i = {
                "id": self.id,
                "n_tokens": len(self.t.encode(s)),
                "description": s,
                "embedding": self.o.embeddings.create(
                    input=[s.replace("\n", " ")], model=self.em
                )
                .data[0]
                .embedding,
            }
            self.id += 1
            return i

        sentences = text.split(". ")
        n_tokens = [len(self.t.encode(" " + sentence)) for sentence in sentences]

        chunks, chunk_id, tokens_so_far, chunk = {}, 0, 0, []
        for sentence, n_token in zip(sentences, n_tokens):

            # a sentence is larger than max tokens
            if n_token > self.mt:
                chunks[chunk_id] = _info(self, sentence)
                chunk_id += 1
                continue

            # accumulated chunks is larger than max tokens
            if tokens_so_far + n_token > self.mt:
                sub_sentence = ". ".join(chunk) + "."
                chunk, tokens_so_far = [], 0
                chunks[chunk_id] = _info(self, sub_sentence)
                chunk_id += 1

            # continue to accumulate chunks
            chunk.append(sentence)
            tokens_so_far += n_token + 1

        # remaining
        sub_sentence = ". ".join(chunk) + "."
        chunks[chunk_id] = _info(self, sub_sentence)

        return chunks

    def break_knowledge(self, fn, fp):
        fn = fn.replace("-", " ").replace("_", " ").replace("#update", "")
        with open(fp, "r", encoding="UTF-8") as f:
            text = f.read()
        text = remove_newlines(text)
        info_chunks = self.split(text)
        file_checksum = self.r.get(self.get_file_key(fp))[:4]
        for chunk_id, chunk_info in info_chunks.items():
            id = f"{self.kp}:{file_checksum}:{chunk_id:05}"
            data = self.get_file_metadata(id, fn, fp) | chunk_info
            self.rp.json().set(id, "$", data)

    def __call__(self):
        floc = self.find()
        for fn, fp in floc.items():
            logging.info(f"embedding {fp}")
            self.break_knowledge(fn, fp)
        results = self.rp.execute()
        return sum(results) == len(results)

    def _verifier(self, id):

        pattern = f"knowledge:0000{id}:*"
        chunk_id_pattern = re.compile(r"knowledge:\d+:(\d+)")

        cursor = "0"
        keys = []
        while cursor != 0:
            cursor, batch = r.scan(cursor=cursor, match=pattern, count=1000)
            keys.extend(batch)

        keys_with_chunks = [
            (key, int(chunk_id_pattern.search(key).group(1))) for key in keys
        ]
        sorted_keys = [key for key, _ in sorted(keys_with_chunks, key=lambda x: x[1])]

        descriptions = []
        for key in sorted_keys:
            value = r.json().get(key)
            if value:
                data = value
                if "description" in data:
                    descriptions.append(data["description"])

        return " ".join(descriptions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings of knowledge")

    parser.add_argument("--redis_host", type=str, default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument(
        "--knowledge_dir",
        type=str,
        help="Directory of the knowledge",
        default="/Users/yiping/Projects/pointer_knowledge/ca/scotia",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to pass the embedding engine",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding engine from OpenAI",
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
    embedder = Embedder(
        r, o, args.embedding_model, args.knowledge_dir, args.max_tokens, args.key_prefix
    )
    success = embedder()
    if success:
        logging.info("embedding successfully")
    else:
        logging.info("embedding not completed")

    # with open("/Users/yiping/Projects/pointer_knowledge/cibc.txt", "r", encoding="UTF-8") as f:
    #     text = f.read()
    # text = remove_newlines(text)
    # print(text)
    # print('====')
    # print(embedder._verifier(0))
    # print(f'verified {text == embedder._verifier(0)}')
