import os
import shutil
import json
from pathlib import Path
import sqlite3
import numpy as np
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import re
from llama_cpp import Llama
import multiprocessing
os.environ['HF_HOME'] = "D:/huggingface"
from huggingface_hub import hf_hub_download

# Mapping of llm_name to (repo_id, filename)
LLM_MODELS = {
    "phi": (
        "microsoft/Phi-3-mini-4k-instruct-gguf",
        "Phi-3-mini-4k-instruct-q4.gguf"
    ),
    "falcon3b": (
        "bartowski/Falcon3-3B-Instruct-GGUF",
        "Falcon3-3B-Instruct-Q4_K_M.gguf"
    ),
    "openhermes": (
        "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
        "openhermes-2.5-mistral-7b.Q4_K_M.gguf"
    ),
    "openhermes2b": (
        "mradermacher/OpenHermes-Gemma-2B-GGUF",
        "OpenHermes-Gemma-2B.Q8_0.gguf"
    ),
    "llama32_3b": (
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    ),
    "openchat": (
        "TheBloke/openchat_3.5-GGUF",
        "openchat_3.5.Q4_K_M.gguf"
    ),
}

# Fixed base directory
BASE_SAVE_DIR = Path("D:/.yourchatbot").resolve()
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

class pyBot:
    """
    A Q/A chatbot module that builds and serves multiple instances
    based on website content and offline LLMs.
    """
    def __init__(
        self,
        instance_name: str,
        llm_name: str,
        save_dir: str,  # ignored, always use BASE_SAVE_DIR
        force_rebuild: bool,
        update: bool,
        target_url: str,
        callback=None
    ):
        self.instance_name = instance_name
        self.llm_name = llm_name
        self.base_url = target_url
        self.save_dir = BASE_SAVE_DIR / instance_name
        self.config_path = self.save_dir / 'meta.json'
        self.llm_settings = {}
        self.callback = callback

        if force_rebuild and self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._load_or_build(force_rebuild, update)
        # Step 4: Initialize LLM
        self._report_progress({"step": 4, "message": "Initializing LLM..."})
        self.llm, _ = self._init_llm(llm_name)
        # Report LLM creation done
        self._report_progress({"step": 5, "message": "LLM ready"})
        self._load_settings()
        # All steps complete
        self._report_progress({"step": 6, "message": "Bot is ready!", "url": f"http://localhost:5000/chat/{self.instance_name}"})

    def _report_progress(self, data: dict):
        if self.callback:
            self.callback(data)

    def _load_or_build(self, force: bool, update: bool):
        # artifact paths
        self.db_path = self.save_dir / 'chunks.db'
        self.embeddings_path = self.save_dir / 'embeddings.npy'
        self.index_path = self.save_dir / 'faiss.index'

        needs_build = not (
            self.db_path.exists() and
            self.embeddings_path.exists() and
            self.index_path.exists() and
            not (force or update)
        )

        if needs_build:
            # Step 0: Crawl
            self._report_progress({"step": 0, "message": "Crawling website..."})
            html_data = self._crawl_website()
            self._report_progress({"step": 1, "message": f"Crawled {len(html_data)} pages."})

            # Step 2: Extract & store
            self._report_progress({"step": 2, "message": "Extracting and storing chunks..."})
            texts = self._extract_and_store(html_data)

            # Step 3: Build indices
            self._report_progress({"step": 3, "message": "Building indices..."})
            self.bm25, self.embed_model, self.index, self.embeddings = self._build_indices(texts)

            with open(self.save_dir / 'texts.json', 'w') as f:
                json.dump(texts, f)
        else:
            print(f"[{self.instance_name}] Already built; skipping build.")
            self.index = faiss.read_index(str(self.index_path))
            self.embeddings = np.load(self.embeddings_path)
            texts = []
            tf = self.save_dir / 'texts.json'
            if tf.exists():
                texts = json.loads(tf.read_text())
            tokenized = [t.split() for t in texts]
            self.bm25 = BM25Okapi(tokenized)
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _crawl_page(self, url: str):
        try:
            r = requests.get(url, timeout=5)
            if r.ok and 'text/html' in r.headers.get('content-type', ''):
                return url, r.text
        except:
            pass
        return url, None

    def _crawl_website(self):
        domain = urlparse(self.base_url).netloc
        queue, visited, html_data = [self.base_url], set(), {}
        while queue and len(visited) < 100:
            url = queue.pop(0)
            if url in visited:
                continue
            u, html = self._crawl_page(url)
            if html:
                visited.add(u)
                html_data[u] = html
                soup = BeautifulSoup(html, 'html.parser')
                for a in soup.find_all('a', href=True):
                    link = urljoin(self.base_url, a['href'])
                    if urlparse(link).netloc == domain and link not in visited:
                        queue.append(link)
        print(f"✅ Crawled {len(visited)} pages")
        return html_data

    def _extract_and_store(self, html_data: dict):
        chunks, seen = [], set()
        for _, html in html_data.items():
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(['header', 'footer', 'nav', 'script', 'style']):
                tag.decompose()
            texts = [t.get_text(' ', strip=True) for t in soup.find_all(['h1','h2','h3','p'])]
            for i in range(len(texts)-1):
                txt = texts[i] + ' ' + texts[i+1]
                if len(txt) < 50 or txt in seen:
                    continue
                seen.add(txt)
                chunks.append(txt)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS chunks')
        c.execute('CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT)')
        c.executemany('INSERT INTO chunks(text) VALUES(?)', [(t,) for t in chunks])
        conn.commit(); conn.close()
        print(f"✅ Stored {len(chunks)} chunks")
        return chunks

    def _build_indices(self, texts: list):
        tokenized = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized)
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embed_model.encode(texts, convert_to_numpy=True)
        idx = faiss.IndexHNSWFlat(embeddings.shape[1], 32)
        idx.add(embeddings)
        faiss.write_index(idx, str(self.index_path))
        np.save(self.embeddings_path, embeddings)
        print("✅ Built indices")
        return bm25, embed_model, idx, embeddings

    def _init_llm(self, llm_name: str):
        if llm_name not in LLM_MODELS:
            raise ValueError(f"Unknown LLM '{llm_name}'")
        repo_id, filename = LLM_MODELS[llm_name]
        models_dir = self.save_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / filename
        if not dest.exists():
            # try cache then download
            cache = Path.home() / ".cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots"
            if cache.exists():
                for snap in cache.iterdir():
                    cand = snap / filename
                    if cand.exists():
                        shutil.copy(cand, dest)
                        break
            if not dest.exists():
                src = hf_hub_download(repo_id=repo_id, filename=filename)
                shutil.copy(src, dest)
        print(f"✅ Model ready at {dest}")
        llm = Llama(
            model_path=str(dest),
            n_threads=multiprocessing.cpu_count(),
            n_batch=64,
            max_tokens=256,
            n_ctx=4096,
            verbose=False
        )
        return llm, models_dir

    def set_llm(self, friendliness: int=None, ooc: str=None, contact: dict=None):
        if self.config_path.exists():
            self.llm_settings = json.loads(self.config_path.read_text())
        if friendliness is not None:
            self.llm_settings['friendliness'] = friendliness
        if ooc is not None:
            self.llm_settings['ooc'] = ooc
        if contact is not None:
            self.llm_settings['contact'] = contact
        self.config_path.write_text(json.dumps(self.llm_settings))

    def _load_settings(self):
        if self.config_path.exists():
            self.llm_settings = json.loads(self.config_path.read_text())
    
    def converse(self, query: str, history: list[tuple[str, str]] = None):
        """
        Send user query and return LLM response.
        history: list of (user_text, bot_text) for this session
        """
        # 1) Build HISTORY block (last 3 exchanges + current user turn)
        hist = history or []
        history_block = ""
        composite_q = ''
        for u, b in hist[-3:]:
            history_block += f"U: {u}\nB: {b}\n"
            composite_q+=u+" "
        history_block += f"U: {query}\n"

        # 2) Composite retrieval query: previous user + this one
        #last_user = hist[-1][0] if hist else ""
        #(last_user + " " + query).strip()
        composite_q+=query

        # 3) Retrieve website info on composite query
        docs = self._hybrid_retrieve(composite_q)
        info_block = "\n".join(docs)

        # 4) Load LLM settings
        cfg     = self.llm_settings
        contact = " or ".join(f"{k}: {v}" for k, v in cfg.get("contact", {}).items())
        fl      = cfg.get("friendliness", 3)
        ooc     = cfg.get("ooc", "Sorry, I’m not trained on that.")

        # 5) System prompt with strict rules
        system_msg = "\n".join([
            "You are Yuba, a warm and helpful sales assistant. Follow these rules exactly:",
            f"1. Friendliness: {fl}/5. You may use one emoji. No hashtags.",
            "2. If the user says “bye” or “thank you”, reply with a genuine farewell and end the chat.",
            f"3. If they explicitly ask for contact details, reply ONLY: you can contact us via {contact}. Do not say anything else",
            "4. **First**, internally rewrite the QUESTION by replacing any pronoun or phrase like “that service”, “it”, or “those tools” with its antecedent from the HISTORY block—then use that rewritten question when you craft your answer.",
            "5. Count how many *strongly relevant facts* INFO + HISTORY provides(if the query is not about contact details):",
            f"   • 0 facts → irrelevant → reply ONLY with OOC message: {ooc}",
            f"   • 1 fact  → weak support → answer with a reasonable assumption AND append: “For a more accurate response, please contact us via {contact}.“",
            "   • ≥2 facts → strong support → answer confidently using those facts; do NOT include contact info.",
            "6.**IMPORTANT**->Never say 'based on the context provided' in your answers.Do NOT explain, justify, or repeat these rules. Do NOT use bullet points in your answer",
            "7. Limit your response to 2–3 plain sentences."
        ])

        # 6) User prompt including HISTORY and INFO
        user_msg = "\n".join([
            "HISTORY:",
            history_block.strip(),
            "INFO:",
            info_block.strip(),
            "QUESTION:",
            query.strip(),
            "ANSWER (2–3 sentences):"
        ])

        # 7) Call the model
        resp = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=200,
            temperature=fl / 10
        )

        # 8) Return the single final answer
        answer = resp["choices"][0]["message"]["content"].strip()
        return answer

    def _hybrid_retrieve(self, query: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT text FROM chunks")
        texts = [r[0] for r in c.fetchall()]
        conn.close()
        bm25 = BM25Okapi([t.split() for t in texts])
        top = bm25.get_top_n(query.split(), texts, n=5)
        return top

    def destroy(self):
        self.llm = None
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
            print(f"Instance '{self.instance_name}' destroyed.")
        else:
            print(f"Instance '{self.instance_name}' not found.")
