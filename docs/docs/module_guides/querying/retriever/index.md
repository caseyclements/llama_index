# Retriever

## Concept

Retrievers are responsible for fetching the most relevant context given a user query (or chat message).

It can be built on top of [indexes](../../indexing/index.md), but can also be defined independently.
It is used as a key building block in [query engines](../../deploying/query_engine/index.md) (and [Chat Engines](../../deploying/chat_engines/index.md)) for retrieving relevant context.

!!! tip
    Confused about where retriever fits in the RAG workflow? Read about [high-level concepts](../../../getting_started/concepts.md)

## Usage Pattern

Get started with:

```python
retriever = index.as_retriever()
nodes = retriever.retrieve("Who is Paul Graham?")
```

## Get Started

Get a retriever from index:

```python
retriever = index.as_retriever()
```

Retrieve relevant context for a question:

```python
nodes = retriever.retrieve("Who is Paul Graham?")
```

> Note: To learn how to build an index, see [Indexing](../../indexing/index.md)

## High-Level API

### Selecting a Retriever

You can select the index-specific retriever class via `retriever_mode`.
For example, with a `SummaryIndex`:

```python
retriever = summary_index.as_retriever(
    retriever_mode="llm",
)
```

This creates a [SummaryIndexLLMRetriever](../../../api_reference/retrievers/summary.md) on top of the summary index.

See [**Retriever Modes**](retriever_modes.md) for a full list of (index-specific) retriever modes
and the retriever classes they map to.

### Configuring a Retriever

In the same way, you can pass kwargs to configure the selected retriever.

> Note: take a look at the API reference for the selected retriever class' constructor parameters for a list of valid kwargs.

For example, if we selected the "llm" retriever mode, we might do the following:

```python
retriever = summary_index.as_retriever(
    retriever_mode="llm",
    choice_batch_size=5,
)
```

## Low-Level Composition API

You can use the low-level composition API if you need more granular control.

To achieve the same outcome as above, you can directly import and construct the desired retriever class:

```python
from llama_index.core.retrievers import SummaryIndexLLMRetriever

retriever = SummaryIndexLLMRetriever(
    index=summary_index,
    choice_batch_size=5,
)
```

## Examples

See more examples in the [retrievers guide](./retrievers.md).
