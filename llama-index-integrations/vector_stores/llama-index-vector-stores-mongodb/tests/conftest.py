import os

import pytest
from llama_index.core import SimpleDirectoryReader
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

import threading
from pathlib import Path

lock = threading.Lock()


@pytest.fixture(scope="session")
def documents(tmp_path_factory):
    """List of documents represents data to be embedded in the datastore.
    Minimum requirements for Documents in the /upsert endpoint's UpsertRequest.
    """
    data_dir = Path(__file__).parents[4] / "docs/docs/examples/data/paul_graham"
    return SimpleDirectoryReader(data_dir).load_data()


@pytest.fixture(scope="session")
def nodes(documents):
    if os.environ.get("OPENAI_API_KEY") is None:
        return None

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=200),
            OpenAIEmbedding(),
        ],
    )

    return pipeline.run(documents=documents)


db_name = os.environ.get("MONGODB_DATABASE", "llama_index_test_db")
collection_name = os.environ.get("MONGODB_COLLECTION", "llama_index_test_vectorstore")
index_name = os.environ.get("MONGODB_INDEX", "vector_index")
MONGODB_URI = os.environ.get("MONGODB_URI")


@pytest.fixture(scope="session")
def atlas_client():
    if MONGODB_URI is None:
        return None

    client = MongoClient(MONGODB_URI)
    assert db_name in client.list_database_names()
    assert collection_name in client[db_name].list_collection_names()
    assert index_name in [
        idx["name"] for idx in client[db_name][collection_name].list_search_indexes()
    ]

    # Clear the collection for the tests
    client[db_name][collection_name].delete_many({})

    return client


@pytest.fixture(scope="session")
def vector_store(atlas_client):
    if MONGODB_URI is None:
        return None

    return MongoDBAtlasVectorSearch(
        mongodb_client=atlas_client,
        db_name=db_name,
        collection_name=collection_name,
        index_name=index_name,
    )


def pytest_addoption(parser):
    """Add CLI options to run tests requiring MONGODB_URI or OPENAI_API_KEY variables."""
    parser.addoption(
        "--mongodb",
        action="store_true",
        help="include tests that require MONGODB_URI environment variable",
    )
    parser.addoption(
        "--openai",
        action="store_true",
        help="include tests that require OPENAI_API_KEY environment variable",
    )


def pytest_collection_modifyitems(config, items):
    """By default, skip tests marked with requires_openai_key or requires_mongodb_uri."""
    if not config.getoption("--mongodb"):
        skip = pytest.mark.skip(
            reason="need --mongodb option to run and MONGODB_URI in os.environ"
        )
        for item in items:
            if item.get_closest_marker("requires_mongodb_uri"):
                item.add_marker(skip)

    if not config.getoption("--openai"):
        skip = pytest.mark.skip(
            reason="need --openai option to run and OPENAI_API_KEY in os.environ"
        )
        for item in items:
            if item.get_closest_marker("requires_openai_key"):
                item.add_marker(skip)
