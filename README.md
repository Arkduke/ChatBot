# Chat Application Using OpenAI and Pinecone

## Overview

This project demonstrates the creation of a chat application using Jupyter Notebook. It utilizes OpenAI's GPT-4o model and Pinecone for retrieval-augmented generation (RAG) to fetch relevant historical chat messages, providing a dynamic conversational AI experience based on past interactions.

## Features

- **OpenAI GPT Model Integration**: Utilizes GPT-4o or other OpenAI models for generating conversational responses.
- **Pinecone Serverless Spec**: Leverages Pinecone for embedding-based storage and retrieval of chat history.
- **Retrieval-Augmented Generation**: Implements RAG to ensure responses are contextually relevant to the conversation history.

## Installation

Before running the application, install the required libraries:

```bash
pip install openai pinecone-client streamlit
