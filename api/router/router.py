from fastapi import APIRouter, Depends, HTTPException
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.models.implementation.hf_chat_model import astra_model
from app.service.inference_service import InferenceService
from app.core.config import settings
from app.repository.chat_repo import ChatRepository
from app.service.memory_service import MemoryService


router = APIRouter()
def get_inference_service() -> InferenceService:
    from motor.motor_asyncio import AsyncIOMotorClient
    
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client["astra"] 
    collection = db["conversations"]
    
    repo = ChatRepository(collection)
    
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
def chat_model(request: ChatRequest):
    """
    Endpoint utama yang akan ditembak oleh microservice PHP kamu 
    setelah user tervalidasi login.
    """

    pesan_user = request.message

    tensor_input = astra_model.preprocess(pesan_user)

    tensor_output = astra_model.predict(tensor_input)

    teks_jawaban = astra_model.postprocess(tensor_output)

    return ChatResponse(
        session_id=request.session_id,
        reply=teks_jawaban
    )
    

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