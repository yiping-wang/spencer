import helper
import argparse


def main(knowledge_dir, embedding_csv, max_tokens, model):
    df = helper.text2df(knowledge_dir)
    shorten_df = helper.shorten(df, max_tokens)
    helper.embed(shorten_df, embedding_csv, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings of knowledge")

    parser.add_argument("--knowledge_dir", type=str, help="Directory of the knowledge")

    parser.add_argument(
        "--embedding_csv", type=str, help="Absoluate pth to the embedding csv"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to pass the embedding engine",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding engine from OpenAI",
    )

    args = parser.parse_args()

    main(args.knowledge_dir, args.embedding_csv, args.max_tokens, args.model)
