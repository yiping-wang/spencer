from setuptools import setup, find_packages

setup(
    name="spencer",
    version="0.1",
    packages=['spencer'],
    install_requires=[
        "openai==1.10.0",
        "tiktoken==0.5.2",
        "pyarrow==15.0.0",
    ],
    # Optional metadata
    author="Yiping Wang",
    author_email="yiping.wang@gmx.com",
    description="A short description of the project.",
    keywords="example project",
    url="https://example.com/project-url",  # Project home page
)
