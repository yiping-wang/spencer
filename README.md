# spencer
based on https://platform.openai.com/docs/tutorials/web-qa-embeddings

# how to create embedding?
1. set openai api key by `export OPENAI_API_KEY=abc`
2. `pip install -r requirements`
3. prepare a directory that stores the knowledge (`*.txt`).
4. `python3 create_embedding.py --knowledge_dir ~/Projects/your_knowledge --embedding_csv ~/Projects/your_knowledge/embedding.csv`

# how to answer question?
1. do "how to create embedding"
2. prepare a question
3. `python3 answer_question.py --embedding_csv ~/Projects/your_knowledge/embedding.csv --question "who is spencer"`

