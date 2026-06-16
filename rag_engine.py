from vector_store import RealEstateVectorStore

class RealEstateRAGEngine:
    def __init__(self, vector_store: RealEstateVectorStore):
        self.store = vector_store

    def generate_response(self, user_prompt):
        if not user_prompt.strip():
            return "Meharbani karkay koi sawal puchiye."

        # Fetch matching records via vector metrics
        matches = self.store.search_similar(user_prompt, top_k=1)

        if matches:
            top_match = matches[0]
            # Threshold verification filter: Check if semantic distance is tightly bound
            if top_match["confidence_score"] < 1.2:
                return top_match["answer"]
        
        # Generative fallback system text when semantic similarity matches fall short
        return "Apki query bohot informative hai! Is mutalik hamare database me exact details nahi hain. Baraye meharbani niche maujood live support or WhatsApp ke zariye hmare consultant se rabta karein."