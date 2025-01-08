from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import openai
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

# Response model
class VitalikResponse(BaseModel):
    reply: str

@router.post("/Vitalik/", response_model=VitalikResponse)
async def vitalik_endpoint(request: VitalikRequest):
    # Retrieve user query and conversation history
    current_message = request.current_message
    previous_messages = request.previous_messages

    # Query Chroma collection
    results = collection.query(
        query_texts=[current_message],
        n_results=1
    )
    
    retrieved_context = results["documents"][0] if results["documents"] else (
        "This isn't something I have a solid answer for at the moment, but it's a fascinating question that might require more exploration or context."
    )

    # Create system prompt
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

    print("DEBUGGING")
    print(f"\n\tretrieved_context - \t{retrieved_context}\n")
    print(f"\n\tconversation_history - \t{conversation_history}\n")

    # Generate response using OpenAI's API
    client = openai.OpenAI(api_key=openai.api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
        )
        ai_response = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

    print(f"\n\tai_response - \t{ai_response}\n")
    return VitalikResponse(reply=ai_response)
