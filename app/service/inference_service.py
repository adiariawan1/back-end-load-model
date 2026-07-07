import asyncio

class InferenceService:
    def __init__(self, model, memory_service):
        self.model = model
        self.memory = memory_service

    async def chat(self, session_id: str, user_message: str, user_name: str = None) -> str:
        """
        Orkestrasi alur penuh: Tarik Memori -> Rakit Prompt -> Generasi AI -> Simpan Memori
        """

        context = await self.memory.get_context(session_id)
        history_lama = context.get("history", [])
        nama_tersimpan = context.get("user_name")

        nama_terdeteksi = await self.memory.remember_name_if_mentioned(session_id, user_message)


        nama_final = user_name or nama_terdeteksi or nama_tersimpan


        if nama_final:
            system_prompt = f"System: Kamu adalah Astra Zero, asisten AI yang cerdas. Kamu sedang berbicara dengan {nama_final}.\n\n"
        else:
            system_prompt = "System: Kamu adalah Astra Zero, asisten AI yang cerdas.\n\n"

        teks_history = ""
        for percakapan in history_lama:
            if percakapan["role"] == "user":
                teks_history += f"User: {percakapan['content']}\n"
            elif percakapan["role"] == "assistant":
                teks_history += f"Astra: {percakapan['content']}\n"
        

        prompt_lengkap = f"{system_prompt}{teks_history}User: {user_message}\n### Response:\n"

        tensor_input = self.model.preprocess(prompt_lengkap)

        tensor_output = await asyncio.to_thread(self.model.predict, tensor_input)

        teks_jawaban = self.model.postprocess(tensor_output)


        await self.memory.save_turn(session_id, user_message, teks_jawaban)


        return teks_jawaban