import gc
import logging
import os
import pickle
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

import faiss
import hdbscan
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ---------------- External Modules ----------------
try:
    from db_update import cluster_update_db
except Exception:
    class cluster_update_db:
        def write_data_DB(self, ids): pass
        def summary_of_updated_clusters(self, ids): pass
        def remove_data_DB(self, ids): pass


# ---------------- Config ----------------
load_dotenv()

BASE_PATH = os.getcwd()

OUTPUT_DIR = os.path.join(BASE_PATH, "enterprise_car_output")
LOG_DIR = os.path.join(BASE_PATH, "enterprise_car_logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "car_intelligence_pipeline.log")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

LAST_DATE_FILE = os.path.join(OUTPUT_DIR, "last_processed_car_date.txt")
CENTROIDS_PKL = os.path.join(OUTPUT_DIR, "car_cluster_centroids.pkl")
INFO_PKL = os.path.join(OUTPUT_DIR, "car_cluster_metadata.pkl")

DB_CONN_STR = os.getenv("ENTERPRISE_CAR_DB_CONNECTION")

TABLE_NAME = "enterprise_vehicle_data_stream"

STREAM_SLEEP = 3600
SIM_THRESHOLD = 0.85


# ---------------- DB ----------------
@contextmanager
def pg_conn():
    conn = psycopg2.connect(DB_CONN_STR)
    try:
        yield conn
    finally:
        conn.close()


def now_utc():
    return datetime.utcnow()


# ---------------- Engine ----------------
class EnterpriseCarClusteringEngine:

    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        dim = self.model.get_sentence_embedding_dimension()

        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.index_ids = set()

        self.cluster_centroids: Dict[int, np.ndarray] = {}
        self.cluster_info: Dict[int, dict] = {}
        self.next_cluster_id = 0

        self.updated_clusters = []

    # ---------------- Persistence ----------------
    def load(self):
        if os.path.exists(CENTROIDS_PKL):
            self.cluster_centroids = pickle.load(open(CENTROIDS_PKL, "rb"))

        if os.path.exists(INFO_PKL):
            self.cluster_info = pickle.load(open(INFO_PKL, "rb"))

        self.index.reset()
        self.index_ids.clear()

        if self.cluster_centroids:
            ids = np.array(list(self.cluster_centroids.keys()), dtype="int64")
            vecs = np.vstack([self.cluster_centroids[i] for i in ids]).astype("float32")
            self.index.add_with_ids(vecs, ids)
            self.index_ids.update(ids.tolist())
            self.next_cluster_id = max(self.cluster_centroids.keys()) + 1

    def save(self):
        pickle.dump(self.cluster_centroids, open(CENTROIDS_PKL, "wb"))
        pickle.dump(self.cluster_info, open(INFO_PKL, "wb"))

    # ---------------- Text Processing ----------------
    def clean_text(self, text: str) -> str:
        text = re.sub(r"<.*?>", " ", str(text))
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip().lower()

    def embed(self, title: str, body: str) -> np.ndarray:
        txt = self.clean_text(f"{title} {body}")
        v = self.model.encode([txt])[0].astype("float32")
        v /= (np.linalg.norm(v) + 1e-8)
        return v

    # ---------------- Clustering ----------------
    def _new_cluster(self, article, emb):
        cid = self.next_cluster_id
        self.next_cluster_id += 1

        self.cluster_centroids[cid] = emb
        self.index.add_with_ids(emb.reshape(1, -1), np.array([cid], dtype="int64"))

        self.cluster_info[cid] = {
            "Vehicles": [article],
            "Brands": set([article.get("brand")]) if article.get("brand") else set(),
            "Sentiments": [article.get("sentiment")],
            "Scores": [article.get("sentimentscore")],
            "LastUpdated": now_utc(),
        }

        return cid

    def assign(self, article):
        emb = self.embed(article.get("title", ""), article.get("text", ""))

        if self.index.ntotal == 0:
            cid = self._new_cluster(article, emb)
            self.updated_clusters.append(cid)
            return

        D, I = self.index.search(emb.reshape(1, -1), 1)
        cid = int(I[0][0])
        score = float(D[0][0])

        if cid in self.cluster_info and score >= SIM_THRESHOLD:
            self.cluster_info[cid]["Vehicles"].append(article)
            self.cluster_info[cid]["LastUpdated"] = now_utc()
            self.cluster_centroids[cid] = (self.cluster_centroids[cid] + emb) / 2
        else:
            cid = self._new_cluster(article, emb)

        self.updated_clusters.append(cid)

    # ---------------- Stream ----------------
    def process_stream(self):
        query = f"""
        SELECT brand, title, text, publish_date, sentiment, sentimentscore
        FROM {TABLE_NAME}
        ORDER BY publish_date DESC
        LIMIT 500;
        """

        with pg_conn() as conn:
            df = pd.read_sql_query(query, conn)

        for row in df.to_dict("records"):
            self.assign(row)


# ---------------- Workers ----------------
def worker():
    engine = EnterpriseCarClusteringEngine()

    while True:
        try:
            engine.load()
            engine.process_stream()
            engine.save()

            if engine.updated_clusters:
                try:
                    db = cluster_update_db()
                    db.write_data_DB(engine.updated_clusters)
                    db.summary_of_updated_clusters(engine.updated_clusters)
                except Exception:
                    logging.exception("DB update failed")

            engine.updated_clusters.clear()
            gc.collect()

        except Exception:
            logging.exception("worker cycle failed")

        time.sleep(STREAM_SLEEP)


# ---------------- Entry ----------------
if __name__ == "__main__":
    t = threading.Thread(target=worker)
    t.start()
    t.join()