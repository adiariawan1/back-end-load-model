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
        """
        Menambahkan satu pesan (user/assistant) ke dalam riwayat percakapan.
        """
        filter_query = {"session_id": session_id}
        new_message = {"role": role, "content": content}
        update_query = {"$push": {"history": new_message}}
        
        await self.collection.update_one(filter_query, update_query, upsert=True)

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