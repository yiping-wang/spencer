# spencer
based on https://platform.openai.com/docs/tutorials/web-qa-embeddings

# install
- naviage to this folder
- `pip install .`
- `import spencer`
- `spencer.create_embedding.main`
- `spencer.answer_question.main`

# how to create embedding?
1. set openai api key by `export OPENAI_API_KEY=abc`
3. prepare a directory that stores the knowledge (`*.txt`). (any file should be fine, but i only tested `.txt`)
5. `python3 create_embedding.py --knowledge_dir ~/Projects/your_knowledge --embedding_csv ~/Projects/your_knowledge/embedding.csv`

# how to answer question?
1. do "how to create embedding"
2. prepare a question
3. `python3 answer_question.py --embedding_csv ~/Projects/your_knowledge/embedding.csv --question "who is spencer"`

