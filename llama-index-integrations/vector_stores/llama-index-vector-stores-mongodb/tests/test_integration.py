"""Integration Tests of llama-index-vector-stores-mongodb
with MongoDB Atlas Vector Datastore and OPENAI Embedding model.

As described in docs/providers/mongodb/setup.md, to run this, one must
have a running MongoDB Atlas Cluster, and
provide a valid OPENAI_API_KEY.
"""

from time import sleep
from llama_index.core import StorageContext, VectorStoreIndex
from .conftest import lock


def test_mongodb_connection(atlas_client, skip_unless_mongodb_uri):
    """Confirm that the connection to the datastore works."""
    assert atlas_client.admin.command("ping")["ok"]


def test_index(
    documents, vector_store, skip_unless_openai_api_key, skip_unless_mongodb_uri
):
    """End-to-end example from essay and query to response.

    via NodeParser, LLM Embedding, VectorStore, and Synthesizer.
    """
    with lock:
        vector_store._collection.delete_many({})
        sleep(2)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context
        )
        query_engine = index.as_query_engine()

        question = "Who is the author of this essay?"
        no_response = True
        response = None
        retries = 5
        search_limit = query_engine.retriever.similarity_top_k
        while no_response and retries:
            response = query_engine.query(question)
            if len(response.source_nodes) == search_limit:
                no_response = False
            else:
                retries -= 1
                sleep(5)
        assert retries
        assert "Paul Graham" in response.response
