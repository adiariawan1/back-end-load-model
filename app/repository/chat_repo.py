class ChatRepository:
    def __init__(self, collection):
        self.collection = collection

    async def find_by_session(self, session_id: str) -> dict:
        """
        Mencari data obrolan berdasarkan ID sesi.
        """
        
        return await self.collection.find_one({"session_id": session_id})

    async def upsert_name(self, session_id: str, name: str):
        """
        Menyimpan atau memperbarui nama user untuk sesi tersebut.
        """
        
        filter_query = {"session_id": session_id}
        update_query = {"$set": {"user_name": name}}
        
        await self.collection.update_one(filter_query, update_query, upsert=True)

    async def append_message(self, session_id: str, role: str, content: str):
        filter_query = {"session_id": session_id}
        update_query = {"$push": {"history": {"role": role, "content": content}}}
        
        # Eksekusi kueri upsert
        result = await self.collection.update_one(filter_query, update_query, upsert=True)
        
        # Cetak jejak kerja database ke terminal
        print(f"DEBUG DB -> Session: {session_id}, Modified: {result.modified_count}, Upserted ID: {result.upserted_id}")

    async def get_recent_history(self, session_id: str, limit: int) -> list:
        """
        Mengambil N pesan terakhir untuk disuapkan sebagai konteks (memory) ke AI.
        """

        filter_query = {"session_id": session_id}

        projection = {"history": {"$slice": -limit}, "_id": 0}
        
        document = await self.collection.find_one(filter_query, projection)
        
    
        if document and "history" in document:
            return document["history"]
        return []