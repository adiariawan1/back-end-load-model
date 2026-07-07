import re

class MemoryService:
    def __init__(self, chat_repository):
        self.repo = chat_repository

    async def get_context(self, session_id: str) -> dict:
        """
        Mengambil konteks utuh (nama dan riwayat) untuk sesi tertentu.
        Jika sesi belum pernah ada, kembalikan dictionary default yang kosong.
        """
        doc = await self.repo.find_by_session(session_id)
        
        if not doc:
            return {"user_name": None, "history": []}
            
        return {
            "user_name": doc.get("user_name"),
            "history": doc.get("history", [])
        }

    async def remember_name_if_mentioned(self, session_id: str, message: str):
        """
        Membaca pesan user, mencari pola penyebutan nama, dan menyimpannya 
        ke database secara otomatis jika terdeteksi.
        """
        pola_regex = r"(?i)(?:nama\s+saya|namaku|panggil\s+saya|panggil\s+saja)\s+([a-zA-Z\s]+)"
        
        pencarian = re.search(pola_regex, message)
        
        if pencarian:
            nama_terdeteksi = pencarian.group(1).strip()

            await self.repo.upsert_name(session_id, nama_terdeteksi)
            
            return nama_terdeteksi
            
        return None

    async def save_turn(self, session_id: str, user_message: str, assistant_reply: str):
        """
        Menyimpan satu putaran penuh percakapan (user dan assistant) secara berurutan.
        """
        await self.repo.append_message(session_id, role="user", content=user_message)
        
        await self.repo.append_message(session_id, role="assistant", content=assistant_reply)