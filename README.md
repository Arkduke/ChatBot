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
```

## Usage
To start the application, run the Streamlit interface:

```bash
streamlit run your_script_name.py
```
## Implementation Details
*API Configuration*
Ensure you have valid API keys from OpenAI and Pinecone. These should be stored securely and not exposed in your code.

```python
OPENAI_API_KEY = 'your-openai-api-key'
PINECONE_API_KEY = 'your-pinecone-api-key'
PINECONE_CLOUD = 'your-pinecone-cloud'
PINECONE_REGION = 'your-pinecone-region'
PINECONE_INDEX_NAME = 'your-pinecone-index-name'
```
## Chat History
The conversation history is stored in an array with each message about 20 tokens long. This history is used to train the model for contextually relevant responses.

## Retrieval-Augmented Generation (RAG)
RAG is implemented to fetch relevant past messages using Pinecone, ensuring that the chatbot's responses are informed by previous interactions.

