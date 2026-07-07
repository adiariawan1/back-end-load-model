from motor.motor_asyncio import AsyncIOMotorClient

from app.core.config import settings

class MongoDB:
    client = None
    db = None
    
    @classmethod
    async def connects(cls):
        if cls.client is None:
            try:
                cls.client = AsyncIOMotorClient(settings.MONGO_URI)
                await cls.client.admin.command('ping')
                
                print("success connect to mongodb")
            except  Exception as e :
                print(f"failed connect to mongodb")
        else :
            print("connection mongodb active")
            
    print(f"🔍 CEK URI: {settings.MONGO_URI}")
    
    @classmethod
    def get_collection(cls, name):
        if cls.db is None:
            cls.db[name]
        else:
            raise Exception("data base not connect")
    
    @classmethod
    def disconnect(cls):
        if cls.client is not None:
            cls.client.close()
            cls.client = None
            cls.db = None
            print("connection is close")
    