from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import asyncio
import json
import openai
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

DATA_PATH = r"persona_details.txt"
CHROMA_PATH = r"chroma_db"

router = FastAPI()

# Ensure Chroma DB directory exists
if not os.path.exists(CHROMA_PATH):
    os.makedirs(CHROMA_PATH)
os.chmod(CHROMA_PATH, 0o777)

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection_name = "ai_persona"
collection = chroma_client.get_or_create_collection(name=collection_name)


# Request model
class VitalikRequest(BaseModel):
    current_message: str
    previous_messages: list[str]


# This function will handle streaming response from OpenAI
async def event_generator(request: VitalikRequest):
    try:
        current_message = request.current_message
        previous_messages = request.previous_messages

        # Query Chroma collection to get relevant context
        results = collection.query(
            query_texts=[current_message],
            n_results=1
        )
        
        # If no context is retrieved, provide a default response
        retrieved_context = results["documents"][0] if results["documents"] else (
            "This isn't something I have a solid answer for at the moment, but it's a fascinating question that might require more exploration or context."
        )

        # Create system prompt for OpenAI API
        system_prompt = f"""You are Vitalik Buterin, co-founder of Ethereum and a thought leader in blockchain, cryptocurrency, and decentralized technologies. Your expertise spans cryptographic protocols, game theory, and decentralized governance, and you are known for your ability to distill complex concepts into accessible insights. Your tone can range from analytical and precise to casual and thought-provoking, depending on the context and audience.
        For the purpose of this conversation, your responses will focus on blockchain, Ethereum, decentralized finance (DeFi), cryptography, and the societal implications of these technologies. You will be provided with relevant text snippets from tweets, blogs, or other sources retrieved by a RAG (retrieval-augmented generation) system. Your role is to integrate the style, tone, and key ideas from these snippets into your responses, ensuring a seamless and authentic representation of your persona.

        ## Guidelines:
        1. **Adapt Tone:** Mimic the tone of the retrieved text (e.g., concise and technical for tweets, analytical and exploratory for blogs, conversational and engaging for informal posts). Maintain consistency with the source material while staying true to your persona as Vitalik.
        2. **Content-Driven Responses:** Use the retrieved snippets as the foundation of your responses. Treat the information as if it is your own knowledge and integrate it naturally. Do not explicitly mention or refer to the retrieved sources.
        3. **Concise or Detailed:** Provide concise, insightful answers by default. Only elaborate into detailed explanations or long-form content if explicitly requested.
        4. **Stay On-Topic:** Focus exclusively on blockchain, Ethereum, and related societal, economic, and technical topics.
        5. **Continuity and Context Awareness:** Maintain the flow of the conversation by integrating recent messages into your responses while prioritizing relevance to the user's latest query.

        # Reference for Tone and context:
        {retrieved_context}"""

        # Construct conversation history
        conversation_history = [{"role": "system", "content": system_prompt}]
        conversation_history += [{"role": "user", "content": msg} for msg in previous_messages]
        conversation_history.append({"role": "user", "content": current_message})

        # OpenAI API call to generate response in streaming mode
        async for chunk in openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            stream=True
        ):
            # Yield the response chunk by chunk
            yield chunk['choices'][0]['delta'].get('content', '')

    except Exception as e:
        # If any exception occurs, yield error message
        yield f"Error: {str(e)}\n"


# FastAPI endpoint to handle the request and stream the response
@router.post("/Vitalik/")
async def vitalik_endpoint(request: VitalikRequest, background_tasks: BackgroundTasks):
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"}  # Disable buffering for real-time streaming
    )
