from functools import partial
import json
import os
from string import Template
from typing import Callable, Dict, Optional

import openai

from guardrails.document_store import DocumentStoreBase, EphemeralDocumentStore
from guardrails.embedding import EmbeddingBase, OpenAIEmbedding
from guardrails.guard import Guard
from guardrails.llm_providers import PromptCallable
from guardrails.utils.sql_utils import create_sql_driver
from guardrails.vectordb import Faiss, VectorDBBase

REASK_PROMPT = """
You are a data scientist whose job is to write SQL queries.

@complete_json_suffix_v2

Here's schema about the database that you can use to generate the SQL query.
Try to avoid using joins if the data can be retrieved from the same table.

{{db_info}}

I will give you a list of examples.

{{examples}}

I want to create a query for the following instruction:

{{nl_instruction}}

For this instruction, I was given the following JSON, which has some incorrect values.

{previous_response}

Help me correct the incorrect values based on the given error messages.
"""


EXAMPLE_BOILERPLATE = """
I will give you a list of examples. Write a SQL query similar to the examples below:
"""


def example_formatter(
    input: str,
    output: str,
    output_schema: Optional[Callable] = None,
) -> str:
    if output_schema is not None:
        output = output_schema(output)

    example = "\nINSTRUCTIONS:\n============\n"
    example += f"{input}\n\n"

    example += "SQL QUERY:\n================\n"
    example += f"{output}\n\n"

    return example


class Text2Sql:
    def __init__(
        self,
        conn_str: str,
        schema_file: Optional[str] = None,
        examples: Optional[Dict] = None,
        embedding: Optional[EmbeddingBase] = None,
        vector_db: Optional[VectorDBBase] = Faiss,
        document_store: Optional[DocumentStoreBase] = EphemeralDocumentStore,
        rail_spec: Optional[str] = None,
        rail_params: Optional[Dict] = None,
        ex_formatter: Optional[Callable] = None,
        reask_prompt: Optional[str] = REASK_PROMPT,
        llm_api: Optional[PromptCallable] = openai.Completion.create,
        num_relevant_examples: int = 2,
    ):
        """Initialize the text2sql application.

        Args:
            conn_str: Connection string to the database.
            schema_file: Path to the schema file. Defaults to None.
            examples: Examples to add to the document store. Defaults to None.
            embedding: Embedding to use for document store. Defaults to OpenAIEmbedding.
            vector_db: Vector database to use for the document store. Defaults to Faiss.
            document_store: Document store to use. Defaults to EphemeralDocumentStore.
            rail_spec: Path to the rail specification. Defaults to "text2sql.rail".
            example_formatter: Fn to format examples. Defaults to example_formatter.
            reask_prompt: Prompt to use for reasking. Defaults to REASK_PROMPT.
        """
        if ex_formatter is None:
            self.example_formatter = partial(
                example_formatter, output_schema=self.output_schema_formatter
            )
        else:
            self.example_formatter = ex_formatter
        self.llm_api = llm_api

        # Initialize the SQL driver.
        self.sql_driver = create_sql_driver(conn=conn_str, schema_file=schema_file)
        self.sql_schema = self.sql_driver.get_schema()

        # Number of relevant examples to use for the LLM.
        self.num_relevant_examples = num_relevant_examples

        # Initialize the Guard class.
        self.guard = self._init_guard(
            conn_str,
            schema_file,
            rail_spec,
            rail_params,
            reask_prompt,
        )

        # Initialize the document store.
        if not embedding:
            embedding = OpenAIEmbedding()
        self.store = self._create_docstore_with_examples(
            examples, embedding, vector_db, document_store
        )

    def _init_guard(
        self,
        conn_str: str,
        schema_file: Optional[str] = None,
        rail_spec: Optional[str] = None,
        rail_params: Optional[Dict] = None,
        reask_prompt: Optional[str] = REASK_PROMPT,
    ):
        # Initialize the Guard class
        if rail_spec is None:
            rail_spec = os.path.join(os.path.dirname(__file__), "text2sql.rail")
            rail_params = {"conn_str": conn_str, "schema_file": schema_file}
            if schema_file is None:
                rail_params["schema_file"] = ""

        # Load the rail specification.
        with open(rail_spec, "r") as f:
            rail_spec_str = f.read()

        # Substitute the parameters in the rail specification.
        if rail_params is not None:
            rail_spec_str = Template(rail_spec_str).safe_substitute(**rail_params)

        guard = Guard.from_rail_string(rail_spec_str)
        guard.reask_prompt = reask_prompt

        return guard

    def _create_docstore_with_examples(
        self,
        examples: Optional[Dict],
        embedder: EmbeddingBase,
        vector_db: VectorDBBase,
        document_store: DocumentStoreBase,
    ) -> Optional[DocumentStoreBase]:
        if examples is None:
            return None

        """Add examples to the document store."""
        if vector_db == Faiss:
            db = Faiss.new_flat_l2_index(embedder.output_dim, embedder=embedder)
        else:
            raise NotImplementedError(f"VectorDB {vector_db} is not implemented.")
        store = document_store(db)
        store.add_texts(
            {example["question"]: {"ctx": example["query"]} for example in examples},
            verbose=True,
        )
        return store

    @staticmethod
    def output_schema_formatter(output) -> str:
        return json.dumps({"generated_sql": output}, indent=4)

    def __call__(self, text: str) -> str:
        """Run text2sql on a text query and return the SQL query."""
        if self.store is not None and self.num_relevant_examples > 0:
            similar_examples = self.store.search(text, self.num_relevant_examples)
            similar_examples_prompt = "\n".join(
                self.example_formatter(example.text, example.metadata["ctx"])
                for example in similar_examples
            )
        else:
            similar_examples_prompt = ""

        try:
            dict_output = self.guard(
                self.llm_api,
                prompt_params={
                    "nl_instruction": text,
                    "examples": similar_examples_prompt,
                    "db_info": str(self.sql_schema),
                },
                max_tokens=512,
            )[1]
            if "generated_sql" in dict_output:
                output = dict_output["generated_sql"]
            else:
                output = list(dict_output.values())[0]
        except TypeError:
            output = None

        return output
