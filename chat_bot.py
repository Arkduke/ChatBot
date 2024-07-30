
from pinecone import Pinecone , ServerlessSpec
from openai import OpenAI
import time
import streamlit as st


# In[3]:


OPENAI_API_KEY = 'xxxxxxxxxxxxxxxxxxxx'
PINECONE_API_KEY = 'xxxxxxxxxxxxxxxxx'
PINECONE_INDEX_NAME = 'chatbox-history'
PINECONE_CLOUD = 'aws'
PINECONE_REGION = 'us-east-1'

MAX_TOKENS = 225


pineconeClient=Pinecone(api_key=PINECONE_API_KEY)
openaiClient = OpenAI(api_key=OPENAI_API_KEY)


# In[4]:


history = [ "1: User: Hi there! How are you doing today? | Bot: Hello! I'm doing great, thank you! How can I assist you today?", "2: User: What's the weather like today in New York? | Bot: Today in New York, it's sunny with a slight chance of rain.", "3: User: Great! Do you have any good lunch suggestions? | Bot: Sure! How about trying a new salad recipe?", "4: User: That sounds healthy. Any specific recipes? | Bot: You could try a quinoa salad with avocado and chicken.", "5: User: Sounds delicious! I'll try it. What about dinner? | Bot: For dinner, you could make grilled salmon with vegetables.", "6: User: Thanks for the suggestions! Any dessert ideas? | Bot: How about a simple fruit salad or yogurt with honey?", "7: User: Perfect! Now, what are some good exercises? | Bot: You can try a mix of cardio and strength training exercises.", "8: User: Any specific recommendations for cardio? | Bot: Running, cycling, and swimming are all excellent cardio exercises.", "9: User: I'll start with running. Can you recommend any books? | Bot: 'Atomic Habits' by James Clear is a highly recommended book.", "10: User: I'll check it out. What hobbies can I take up? | Bot: You could explore painting, hiking, or learning a new instrument.", "11: User: Hiking sounds fun! Any specific trails? | Bot: There are great trails in the Rockies and the Appalachian Mountains.", "12: User: I'll plan a trip. What about indoor activities? | Bot: Indoor activities like reading, cooking, or playing board games.", "13: User: Nice! Any good board games? | Bot: Settlers of Catan and Ticket to Ride are both excellent choices.", "14: User: I'll try them out. Any movie recommendations? | Bot: 'Inception' and 'The Matrix' are must-watch movies.", "15: User: I love those movies! Any TV shows? | Bot: 'Breaking Bad' and 'Stranger Things' are very popular.", "16: User: Great choices! What about podcasts? | Bot: 'How I Built This' and 'The Daily' are very informative.", "17: User: Thanks! What are some good travel destinations? | Bot: Paris, Tokyo, and Bali are amazing travel spots.", "18: User: I'll add them to my list. Any packing tips? | Bot: Roll your clothes to save space and use packing cubes.", "19: User: That's helpful! What about travel insurance? | Bot: Always get travel insurance for safety and peace of mind.", "20: User: Thanks for the tips! Any last advice? | Bot: Just enjoy your journey and make the most out of your experiences." ]


# In[5]:

st.header('OpenAI Chatbot')
User_prompt = st.text_area(f'Enter your query here')


def add_embeddings_to_pinecone(history, index_name=PINECONE_INDEX_NAME):
    existing_indexes = pineconeClient.list_indexes()
    index_exists = any(index['name'] == index_name for index in existing_indexes)

    if index_exists:
        print(f"Index '{index_name}' already exists.")

    else:
        print(f"Creating index '{index_name}'")
        pineconeClient.create_index(
            index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        print("Index created.")
    data_embedding = [openaiClient.embeddings.create(input = message, model="text-embedding-3-small").data[0].embedding for message in history]
    vectors = [{"id": f"message-{i+1}", "values": embedding, "metadata": {"text": history[i]}} for i, embedding in enumerate(data_embedding)]
    pineconeClient.Index(PINECONE_INDEX_NAME).upsert(vectors)
    time.sleep(5)


# In[6]:


def retrieve_relevant_history(query, history, index_name):
    add_embeddings_to_pinecone(history,index_name)
    query_embeddings = openaiClient.embeddings.create(input = [query], model="text-embedding-3-small").data[0].embedding
    retrieved_data = pineconeClient.Index(PINECONE_INDEX_NAME).query(
        vector=query_embeddings,
        top_k=3,
        include_metadata=True
    )  
    retrieved_answer = [match['metadata'] for match in retrieved_data['matches']]
    return retrieved_answer
    # return retrieved_answer[0]['text']




# In[7]:


def prepare_prompt(test_prompt,history,index_name):
    relevant_history_dic = retrieve_relevant_history(test_prompt, history, index_name)
    relevant_history = [item['text'] for item in relevant_history_dic]

    context = "".join(relevant_history)
    combined_prompt = [
            {"role": "system", "content": "You are a chatbot which gives answers by analyzing the provided chats between the bot and the user. If the context doesn't match, please provide a polite answer you are unable to answer."},
            {"role": "assistant", "content": context},
            {"role": "user", "content": test_prompt}
            ] 
    if len(str(combined_prompt).split()) > MAX_TOKENS:
        combined_prompt = "".join(str(combined_prompt).split()[:MAX_TOKENS])
        
    return combined_prompt, relevant_history


# In[8]:


def test_final_prompt(final_test_prompt):
    # final_test_prompt = "Do you think it will help me stay fit?"
    prepared_prompt, context_referred = prepare_prompt(final_test_prompt,history,PINECONE_INDEX_NAME)   
    response = openaiClient.chat.completions.create(
        model ="gpt-4o-mini",
        messages = prepared_prompt
    )    
    # print(prepared_prompt)
    print(f"Final Test Prompt: {final_test_prompt}")
    print(f"Context Referred: {context_referred}")
    print(f"Final Test Prompt Response: {response.choices[0].message.content}")
    st.write(response.choices[0].message.content)


if st.button('Submit'):
    test_final_prompt(User_prompt)




