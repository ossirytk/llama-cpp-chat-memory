import importlib.util
from typing import Any

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.schema.embeddings import Embeddings


class CustomSpacyEmbeddings(BaseModel, Embeddings):
    model_path: str
    nlp: Any

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: dict) -> dict:
        model_path = values["model_path"]
        # Check if the Spacy package is installed
        if importlib.util.find_spec("spacy") is None:
            spacy_not_installed_error_message = "Spacy package not found. Please install it with `pip install spacy`."
            raise ValueError(spacy_not_installed_error_message)
        try:
            # Try to load the 'en_core_web_sm' Spacy model
            import spacy

            values["nlp"] = spacy.load(model_path)
        except OSError:
            # If the model is not found, raise a ValueError
            error_message = f"""Spacy model not found.
                Please install it with
                python -m spacy download {model_path}"""
            raise ValueError(error_message) from None
        return values  # Return the validated values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): The documents to generate embeddings for.

        Returns:
            A list of embeddings, one for each document.
        """
        return [self.nlp(text).vector.tolist() for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """
        Generates an embedding for a single piece of text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            The embedding for the text.
        """
        return self.nlp(text).vector.tolist()
