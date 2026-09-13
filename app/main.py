from fastapi import FastAPI
from contextlib import asynccontextmanager


from api.router.router import router as chat_router
from app.core.config import settings

from app.core.database import MongoDB 
from app.models.implementation.hf_chat_model import astra_model

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("start server")

    print("Data Base Connected")


    print("load model")
    astra_model.load(settings.HF_MODEL_ID)

    app.state.astra_model = astra_model

    print("ready")
    yield 


    print("stop")
    
    
    print("server off")



app = FastAPI(title="Astra Zero AI Chat Service", lifespan=lifespan)

app.include_router(chat_router)
