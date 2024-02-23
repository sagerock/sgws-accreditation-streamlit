import openai
import tiktoken
import numpy as np  
import os
import streamlit as st
import json
from streamlit_chat import message
import pinecone
import random

from PIL import Image

# When you are working locally set your api keys with this:
# openai.api_key = os.getenv('OPENAI_API_KEY')
# pinecone_api_key = os.getenv('PINECONE_API_KEY')

# When you are uploading to Streamlit, set your keys like this:
pinecone_api_key = st.secrets["API_KEYS"]["pinecone"]
openai.api_key = st.secrets["API_KEYS"]["openai"]

pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp")


logo = Image.open('images/logo.jpg')



# random user picture
# user_av = random.randint(0, 100)
user_av = 5

# random bott picture
# bott_av = random.randint(0, 100)
bott_av = 10

def randomize_array(arr):
    sampled_arr = []
    while arr:
        elem = random.choice(arr)
        sampled_arr.append(elem)
        arr.remove(elem)
    return sampled_arr

st.set_page_config(page_title="Spring Garden Waldorf School Accreditation AI Chatbot", layout="wide")

st.header("Do you have a question about Spring Garden Waldorf School Accreditation? I might be able to help. \n")



# st.header("Thank you for visiting the Spring Garden Waldorf School Accreditation AI Chatbot \n")

# Define the name of the index and the dimensionality of the embeddings
index_name = "sagerock"
dimension = 1536

pineconeindex = pinecone.Index(index_name)


######################################
#######
#######   OPEN AI SETTINGS !!!
#######
#######
######################################


#COMPLETIONS_MODEL = "text-davinci-003"
COMPLETIONS_MODEL = "gpt-4-0125-preview"
EMBEDDING_MODEL = "text-embedding-ada-002"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,  
    "max_tokens": 4000,
    "model": COMPLETIONS_MODEL,
}


main_url = "https://www.sgws.org"


with st.sidebar:
    st.image('images/logo.jpg', width=250)
    st.markdown("---")

    st.markdown("Welcome to the Spring Garden Waldorf School Accreditation Chatbot! \n")
    st.markdown(
        
"This chatbot has the complete knowledge of the Spring Garden Waldorf School Accreditation. \n"


"Now, go ahead and ask any question you have about the Spring Garden Waldorf School Accreditation! \n"

    )


# MAIN FUNCTIONS




def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



def get_embedding(text, model):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]



MAX_SECTION_LEN = 2500 #in tokens
SEPARATOR = "\n"
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))



def construct_prompt_pinecone(question):
    """
    Fetch relevant information from pinecone DB
    """
    xq = get_embedding(question , EMBEDDING_MODEL)

    #print(xq)

    res = pineconeindex.query([xq], top_k=30, include_metadata=True, namespace="sgws-accred")

    #print(res)
    # print(most_relevant_document_sections[:2])

    chosen_sections = []    
    chosen_sections_length = 0

    for match in res['matches'][:12]:
        #print(f"{match['score']:.2f}: {match['metadata']['text']}")
        if chosen_sections_length <= MAX_SECTION_LEN:
            document_section = match['metadata']['text']

            #   document_section = str(_[0] + _[1])      
            chosen_sections.append(SEPARATOR + document_section)

            chosen_sections_length += num_tokens_from_string(str(document_section), "gpt2")

    for match in randomize_array(res['matches'][-18:]):
        #print(f"{match['score']:.2f}: {match['metadata']['text']}")
        if chosen_sections_length <= MAX_SECTION_LEN:
            document_section = match['metadata']['text']

            #   document_section = str(_[0] + _[1])      
            chosen_sections.append(SEPARATOR + document_section)

            chosen_sections_length += num_tokens_from_string(str(document_section), "gpt2")


    # Useful diagnostic information
    #print(f"Selected {len(chosen_sections)} document sections:")
    
    header = """Welcome to the Spring Garden Waldorf School Accreditation Chatbot! \n

This chatbot has the complete knowledge of the Spring Garden Waldorf School Accreditation.\n"

"Now, go ahead and ask any question you have about the Spring Garden Waldorf School Accreditation! \n

Context: Spring Garden Waldorf School Accreditation \n
    """ 
    return header + "".join(chosen_sections) 



#TO BE ADDED: memory with summary of past discussions

def summarize_past_conversation(content):

    APPEND_COMPLETION_PARAMS = {
        "temperature": 0.0,
        "max_tokens": 300,
        "model": COMPLETIONS_MODEL,
    }

    prompt = "Please respond to the question asked thoroughly and format it with markdown: Only responde to questions based on the content in your database of Spring Garden Waldorf School Accreditation. If you don't know the answer just let them know that you don't now and the person should reach out to the Spring Garden Waldorf School Accreditation for more information.\n" + content

    try:
        response = openai.Completion.create(
                    prompt=prompt,
                    **APPEND_COMPLETION_PARAMS
                )
    except Exception as e:
        print("I'm afraid your question failed! This is the error: ")
        print(e)
        return None

    choices = response.get("choices", [])
    if len(choices) > 0:
        return choices[0]["text"].strip(" \n")
    else:
        return None





COMPLETIONS_API_PARAMS = {
        "temperature": 0.7,
        "max_tokens": 500,
        "model": COMPLETIONS_MODEL,
    }


def answer_query_with_context_pinecone(query):
    prompt = construct_prompt_pinecone(query) + "\n\n Q: " + query + "\n A:"
    
    print("---------------------------------------------")
    print("prompt:")
    print(prompt)
    print("---------------------------------------------")
    try:
        response = openai.ChatCompletion.create(
                    messages=[{"role": "system", "content": "You are a highly knowledgeable chatbot that knows a great deal about Spring Garden Waldorf School Accreditation. Please only respond to questions from the information provided. If you don't know the answer just let the person know that you don't know the answer and they should reach out to Spring Garden Waldorf School for more information."},
                            {"role": "user", "content": str(prompt)}],
                            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                            # {"role": "user", "content": "Where was it played?"}
                            # ]
                    **COMPLETIONS_API_PARAMS
                )
    except Exception as e:
        print("I'm afraid your question failed! This is the error: ")
        print(e)
        return None

    choices = response.get("choices", [])
    if len(choices) > 0:
        return choices[0]["message"]["content"].strip(" \n")
    else:
        return None



# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def clear_text():
    st.session_state["input"] = ""

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("Input a question here! For example: \"Tell me about the Spring Garden Waldorf School Accreditation\". \n Also, I have no memory of previous questions!😊")
    return input_text



user_input = get_text()


if user_input:
    output = answer_query_with_context_pinecone(user_input)

    # store the output 
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Display chatbot responses
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display chatbot response in Markdown format
        st.markdown(st.session_state['generated'][i])
        
        # Display user input
        message(st.session_state['past'][i], is_user=True, avatar_style="personas", seed=user_av, key=str(i) + '_user')