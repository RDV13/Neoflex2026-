from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import faiss
from typing import List, Dict, Tuple

class HybridRetriever:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        # Семантический поиск
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.embeddings = None

        # Лексический поиск
        self.bm25 = None
        self.documents = []
        self.tokenized_corpus = []  # Храним токенизированный корпус

    def add_documents(self, texts: List[str], metadata: List[Dict] = None):
        """Добавляет документы в ретривер"""
        self.documents.extend(texts)

        # Токенизируем новые документы
        new_tokenized = [doc.split(" ") for doc in texts]
        self.tokenized_corpus.extend(new_tokenized)

        # Перестраиваем BM25 
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Обновляем векторный индекс
        new_embeddings = self.embedding_model.encode(texts).astype('float32')

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # Нормализуем все векторы для косинусного сходства
        faiss.normalize_L2(self.embeddings)

        # Создаём FAISS индекс
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Выполняет гибридный поиск"""
        # Лексический поиск (BM25)
        bm25_scores = self.bm25.get_scores(query.split(" "))
        bm25_top_k = np.argsort(bm25_scores)[-k:]

        # Семантический поиск (векторный)
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embedding)  # Нормализуем запрос
        similarities, indices = self.index.search(
            query_embedding.astype('float32'), k
        )

        # Гибридная оценка: среднее нормализованных оценок
        hybrid_scores = {}
        max_bm25 = max(bm25_scores) if bm25_scores else 1
        for idx, score in enumerate(bm25_scores):
            normalized_score = score / max_bm25 if max_bm25 > 0 else 0
            hybrid_scores[idx] = normalized_score

        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx in hybrid_scores:
                hybrid_scores[idx] = (hybrid_scores[idx] + sim) / 2
            else:
                hybrid_scores[idx] = sim

        # Сортируем и берём топ-k
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for doc_idx, score in sorted_results:
            results.append({
                'text': self.documents[doc_idx],
                'score': float(score),
                'source_id': doc_idx
            })
        return results
