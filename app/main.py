from fastapi import FastAPI
from contextlib import asynccontextmanager


from api.router.router import router as chat_router
from app.core.config import settings

from app.core.database import MongoDB 
from app.models.implementation.hf_chat_model import astra_model

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("🚀 Memulai inisialisasi server...")

    print("✅ Database terhubung.")


    print("🧠 Membangunkan arsitektur Astra Zero...")
    astra_model.load(settings.HF_MODEL_ID)

    app.state.astra_model = astra_model

    print("🔥 Astra Zero Microservice siap menerima request!")
    yield 


    print("🛑 Sinyal berhenti diterima. Membersihkan memori...")
    
    
    print("💤 Server berhasil dimatikan dengan aman.")



app = FastAPI(title="Astra Zero AI Chat Service", lifespan=lifespan)

app.include_router(chat_router)