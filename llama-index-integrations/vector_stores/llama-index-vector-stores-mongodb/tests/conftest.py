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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def skip_unless_openai_api_key():
    # if "OPENAI_API_KEY" not in os.environ:
    if OPENAI_API_KEY is None:
        pytest.skip("Test requires OPENAI_API_KEY in os.environ")


@pytest.fixture(scope="session")
def documents(tmp_path_factory):
    """List of documents represents data to be embedded in the datastore.
    Minimum requirements for Documents in the /upsert endpoint's UpsertRequest.
    """
    data_dir = Path(__file__).parents[4] / "docs/docs/examples/data/paul_graham"
    return SimpleDirectoryReader(data_dir).load_data()


@pytest.fixture(scope="session")
def nodes(documents):
    if OPENAI_API_KEY is None:
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
MONGODB_URI = os.environ.get("MONGODB_URI", "")


@pytest.fixture(autouse=True)
def skip_unless_mongodb_uri():
    # if "OPENAI_API_KEY" not in os.environ:
    if MONGODB_URI is None:
        pytest.skip("Test requires MONGODB_URI in os.environ")


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
