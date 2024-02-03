from setuptools import setup, find_packages

setup(
    name="spencer",
    version="0.2",
    packages=['spencer'],
    install_requires=[
        "openai",
        "tiktoken",
        "pyarrow",
        "redis",
        "redisearch"
    ],
    # Optional metadata
    author="Yiping Wang",
    author_email="yiping.wang@gmx.com",
    description="A short description of the project.",
    keywords="example project",
    url="https://example.com/project-url",  # Project home page
)
