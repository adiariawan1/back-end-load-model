from fastapi import APIRouter, Depends, HTTPException
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.models.implementation.hf_chat_model import astra_model
from app.service.inference_service import InferenceService
from app.core.config import settings
from app.repository.chat_repo import ChatRepository
from app.service.memory_service import MemoryService
from motor.motor_asyncio import AsyncIOMotorClient

router = APIRouter()


mongo_client = AsyncIOMotorClient(settings.MONGO_URI)
mongo_db = mongo_client["astra"] 
chat_collection = mongo_db["conversations"]


async def get_inference_service() -> InferenceService:
    # Gunakan collection global
    repo = ChatRepository(chat_collection)
    memory = MemoryService(repo)
    return InferenceService(model=astra_model, memory_service=memory)


@router.get("/health")
def health_check():
    """
    Endpoint sederhana untuk pengecekan berkala (health check) dari microservice PHP 
    atau load balancer untuk memastikan engine AI ini masih hidup.
    """
    return {"status": "ok", "service": "Astra Zero AI Chat Engine"}

@router.post("/chat", response_model=ChatResponse)
async def chat_model(
    request: ChatRequest,
    inference_service: InferenceService = Depends(get_inference_service)
):
    """
    Endpoint utama yang akan ditembak oleh microservice PHP.
    """
    try:
        reply = await inference_service.chat(
            session_id=request.session_id,
            user_message=request.message,
            user_name=request.user_name if hasattr(request, 'user_name') else None
        )
        
        return ChatResponse(
            session_id=request.session_id,
            reply=reply
        )
    except Exception as e:
        print(f"❌ Error saat memproses chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/chat/{session_id}")
async def get_chat_history(
    session_id: str,
    inference_service: InferenceService = Depends(get_inference_service)
):
    """
    Endpoint untuk mengambil riwayat obrolan masa lalu berdasarkan ID Sesi.
    """
    try:
        context = await inference_service.memory.get_context(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "user_name": context.get("user_name"),
            "history": context.get("history")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))