from sentence_transformers import SentenceTransformer, util
import json

class DreamJobModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        with open("jobs.json", "r") as f:
            self.jobs = json.load(f)
        self.job_embeddings = self.model.encode([job['desc'] for job in self.jobs])

    def predict_job(self, user_text):
        user_embedding = self.model.encode(user_text)
        similarities = util.cos_sim(user_embedding, self.job_embeddings)[0]
        best_match_idx = similarities.argmax()
        return self.jobs[best_match_idx]
