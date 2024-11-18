import os
import sys
from time import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
from openai import OpenAI

from data import Data
from helper import Embedder, NER


# Setup
model_path = None
device_map = "cpu"
if len(sys.argv) > 1: model_path = sys.argv[1] # "stella_en_400M_v5"
if len(sys.argv) > 2: device_map = sys.argv[2] # "mps"
if model_path is None:
    st.error("Please provide the `model_path` argument.")
    sys.exit(1)
if "OPENAI_API_KEY" not in os.environ:
    st.error("Please set the `OPENAI_API_KEY` environment variable.")
    sys.exit(1)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
materials = Data()
embedder = Embedder(model_path=model_path, device_map=device_map)
ner = NER(device_map=device_map)

# Bonus audio replies
st.markdown("Re: I want to ask Scott Manley about his favorite things to do in Kerbal Space Program.")
with open("audio/response_Q1.wav", "rb") as audio_file: st.audio(audio_file.read(), format="audio/mp3")
st.markdown("Re: What did Fireship say about Apache Kafka?")
with open("audio/response_Q2.wav", "rb") as audio_file: st.audio(audio_file.read(), format="audio/mp3")
st.markdown("Re: What did the July 6th, 2022 Code Report video discuss?")
with open("audio/response_Q3.wav", "rb") as audio_file: st.audio(audio_file.read(), format="audio/mp3")

# Initialize the chat history in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Talk with Experts")

# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your message here..."):
    # Append the user's message to the session state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the assistant's response
    with st.chat_message("assistant"):
        start = time()
        prompt_vector = embedder.embed(prompt)

        # Retrieve the expert with the most similar background
        similarities_intro = embedder.similarity(prompt_vector, materials.expert_intro_vectors)
        idx = int(np.argmax(similarities_intro))
        expert = list(materials.expert_intro[int(idx)].keys())[0]

        # Retrieve the most relevant publications of that expert
        similarities_publication = embedder.similarity(prompt_vector, materials.publication_vectors[expert])
        publication_ranked = materials.publication[expert].iloc[np.argsort(-similarities_publication[0])]
        content = publication_ranked["content_text"].values.tolist()
        reference = publication_ranked["content_source_url"].values.tolist() 
        entities, query_dates = ner.run(prompt)
        retrivals = []
        for c, r in zip(content, reference): # start from the most relevant publication and descend
            if len(retrivals) == 5: break # at most get 5 retrivals
            keywords = []; date_to_match = True if query_dates else False
            retrival_dates = list(set([ner.standardize_date(date) for date in ner.date_regex.findall(c)]))
            if date_to_match: # if there are dates in the query
                for date in query_dates:
                    for date_ref in retrival_dates:
                        if date.lower() in date_ref.lower():
                            keywords.append(date)
                if len(keywords) == 0: continue # if the retrival does not contain the date in the query, skip
            for e in entities:
                if e["text"].lower() in c.lower():
                    keywords.append(e["text"])
            if len(keywords) > 0:
                retrivals.append({"content": c, "reference": r, "keywords": keywords})

        # Check the relationship if no keywords found in the retrivals
        if len(retrivals) == 0:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "user", "content": f"{content[0]}\n\nIs the above content highly helpful for answering the below query? Just answer yes or no without explanation:\n{prompt}"},
                ]
            )
            response = completion.choices[0].message.content
            if "yes" in response.lower():
                retrivals = [{"content": content[0], "reference": reference[0], "keywords": []}]
        
        # Prepare the messages for the assistant
        intro = ""
        for item in materials.expert_intro:
            name, = item
            if name == expert:
                intro = item[name]
                break
        system = f"You are acting as a specific expert: {intro}\n\nPlease try to reply using the expert's speech style and knowledge." if intro else ""
        retrivals_ref = ""
        if len(retrivals) > 0:
            retrivals_text = "\n\n".join([retrieval["content"] for retrieval in retrivals])
            retrivals_ref = "\n\nPlease kindly check out my videos here:\n" + "\n".join([f"{idx+1}. {retrieval['reference']}" for idx, retrieval in enumerate(retrivals)])
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{retrivals_text}\n\nPlease try to answer the question below based on the above references:\n{prompt}"},
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{prompt}\n\nThere is no references for this question. Please try to answer the question by your knowledge. If you don't know the answer, just say 'Sorry, there is no enough information to answer your question.' without any make-up."},
            ]

        assistant_response = ""
        message_placeholder = st.empty()

        # Get the response from OpenAI API with streaming
        stream = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages, # st.session_state.messages,
            stream=True,
        )

        # Stream the assistant's response
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                assistant_response += content
                message_placeholder.markdown(assistant_response + "▌")

        if retrivals_ref:
            assistant_response += retrivals_ref

        message_placeholder.markdown(assistant_response)

        # Append the assistant's response to the session state
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        print("-" * 32)
        print(f"Completed response in {time() - start:.2f}s")
        print("-" * 32)
