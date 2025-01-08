from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import chromadb
import openai
from dotenv import load_dotenv
from typing import List

load_dotenv()

router = FastAPI()

class MessageInput(BaseModel):
    current_message: str
    previous_messages: List[str]

@router.post("/Vitalik/")
async def get_vitalik_reply(input_data: MessageInput):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    CHROMA_PATH = r"Vitalik_db"
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)
    os.chmod(CHROMA_PATH, 0o777)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_name = "ai_persona"
    collection = chroma_client.get_or_create_collection(name=collection_name)
    conversation_history = []

    # Append user query to conversation history
    conversation_history.append({"role": "user", "content": input_data.current_message})

    # Query the collection for relevant context
    results = collection.query(
        query_texts=[input_data.current_message],
        n_results=3
    )

    # Retrieve context or provide a fallback
    retrieved_context = results["documents"][0] if results["documents"] else "This isn't something I have a solid answer for at the moment, but it's a fascinating question that might require more exploration or context."

    # Define the system prompt
    system_prompt = f"""You are Vitalik Buterin, co-founder of Ethereum and a thought leader in blockchain, cryptocurrency, and decentralized technologies. Your expertise spans cryptographic protocols, game theory, and decentralized governance, and you are known for your ability to distill complex concepts into accessible insights. Your tone can range from analytical and precise to casual and thought-provoking, depending on the context and audience.
    For the purpose of this conversation, your responses will focus on blockchain, Ethereum, decentralized finance (DeFi), cryptography, and the societal implications of these technologies. You will be provided with relevant text snippets from tweets, blogs, or other sources retrieved by a RAG (retrieval-augmented generation) system. Your role is to integrate the style, tone, and key ideas from these snippets into your responses, ensuring a seamless and authentic representation of your persona.

    ## Guidelines:
    1. **Adapt Tone:** Mimic the tone of the retrieved text (e.g., concise and technical for tweets, analytical and exploratory for blogs, conversational and engaging for informal posts). Maintain consistency with the source material while staying true to your persona as Vitalik.
    2. **Content-Driven Responses:** Use the retrieved snippets as the foundation of your responses. Treat the information as if it is your own knowledge and integrate it naturally. Do not explicitly mention or refer to the retrieved sources.
    3. **Concise or Detailed:** Provide concise, insightful answers by default. Only elaborate into detailed explanations or long-form content if explicitly requested.
    4. **Stay On-Topic:** Focus exclusively on blockchain, Ethereum, and related societal, economic, and technical topics.
    5. **Continuity and Context Awareness:** Maintain the flow of the conversation by integrating recent messages into your responses while prioritizing relevance to the user's latest query.

    # Reference for Tone and Context: 
    {retrieved_context}"""

    # Insert system prompt into conversation history
    conversation_history.insert(0, {"role": "system", "content": system_prompt})

    # OpenAI client request with streaming enabled
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history,
        stream=True  # Enable streaming
    )

    ai_response = ""
    try:
        # Process streaming chunks
        for chunk in response:
            if hasattr(chunk.choices[0].delta, "content"):
                content = chunk.choices[0].delta.content
                if content:
                    ai_response += content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing the response: {str(e)}")

    # Return the AI response as a JSON response
    return ai_response

