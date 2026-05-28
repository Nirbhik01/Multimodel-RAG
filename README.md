# Multimodel-RAG

AI-powered medical diagnostic assistant that combines chest X-ray images and clinical text to retrieve similar historical cases and provide explainable radiological insights. Uses a multimodal Retrieval-Augmented Generation (RAG) pipeline with hybrid vector + sparse retrieval and local LLM inference.

## Project Structure

```text
root/
├── backend/
│   ├── api/
│   │   ├── apps.py               # Django app config — loads models at startup
│   │   ├── views.py              # Main RAG endpoint + RRF ranking
│   │   └── mongodb_utils.py      # Conversation history helpers
│   ├── core/
│   │   ├── embedding/
│   │   │   ├── medical_models.py # PubMedBERT + Rad-DINO singletons
│   │   │   └── generate_embeddings.py
│   │   ├── retrieval/
│   │   │   ├── query_vector_db.py
│   │   │   ├── bm25_index.py
│   │   │   └── cross_encoder_reranker.py
│   │   └── generation/
│   │       └── generate_answer.py
│   └── .env.example
├── data/
│   ├── images/
│   │   └── images_normalized/    # 1000+ PNG chest X-ray files
│   ├── processed/                # embedding_input.json, embedding_results.json
│   ├── raw/                      # indiana_projections.csv, indiana_reports.csv
│   └── test_images/
├── frontend/
├── ingestion/
│   ├── generate_medical_embeddings.py
│   ├── upload_vectors.py
│   └── .env.example
├── utils/
│   ├── remove_unwanted_data.ipynb
│   ├── link_images_and_report.ipynb
│   ├── IU_data_cleaning_and_classification.ipynb
│   ├── preprocess_text.ipynb
│   └── knowledge_base_to_vector.ipynb
├── requirements.txt
└── README.md
```

## Architecture

### Embedding Models (768-dim each)
- **Text**: PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`) via SentenceTransformer
- **Image**: Rad-DINO (`microsoft/rad-dino`) via HuggingFace Transformers, uses `[CLS]` token
- Both models load as singletons at Django startup (`api/apps.py`) to avoid slow first-request latency

### Retrieval Pipeline
1. Generate text embedding (PubMedBERT) and image embedding (Rad-DINO) for the query
2. Query Upstash Vector DB with `type == "text"` and `type == "image"` filters separately (top-10 each)
3. Run BM25 sparse retrieval on query text
4. Combine all three result sets via Reciprocal Rank Fusion (RRF, k=60)
5. Cross-encoder reranking of RRF top-10 pool by scoring (query, cleaned_text) pairs
6. Select top-3 cases for LLM context (falls back to RRF top-3 on image-only queries)

### LLM Generation (Ollama)
- Local LLM via Ollama, model configurable via `OLLAMA_MODEL` env var (default: `qwen2`)
- Vision-capable: receives patient image + top-3 reference case images (compressed 512×512 JPEG, quality=85)
- Single-pass inference: all candidates sent in one prompt with anonymized labels ("Reference Case 1", etc.)
- GPU auto-detection via torch CUDA check with nvidia-smi fallback
- Responses streamed via `StreamingHttpResponse` with newline-delimited JSON

### Storage
- **Upstash Vector DB**: two vectors per record — `img_{id}` (type=image) and `txt_{id}` (type=text) — with metadata: `original_id`, `label`, `cluster`, `normal`, `cleaned_text`
- **MongoDB**: conversation history (database: `multimodal_rag`, collection: `conversations`); messages store base64-encoded images
- **BM25 index**: built at startup from `data/processed/embedding_input.json` using `cleaned_text` field

## Setup

**Prerequisites**: Python 3.9+, Node.js 16+, Ollama, MongoDB, Upstash Vector account

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r ../requirements.txt
cp .env.example .env           # fill in values
python manage.py runserver 0.0.0.0:8000

# Frontend
cd frontend
npm install
cp .env.example .env           # set VITE_API_BASE_URL
npm run dev                    # http://localhost:5173

# Ollama
ollama serve
ollama pull qwen2
```

## Environment Variables

**Backend** (`backend/.env`):

| Variable | Description |
|----------|-------------|
| `UPSTASH_DB_URL` | Upstash Vector REST endpoint |
| `UPSTASH_TOKEN` | Upstash write/read token |
| `UPSTASH_READ_ONLY_TOKEN` | Upstash read-only token |
| `MONGO_DB_CONNECTION_STRING` | MongoDB connection string |
| `FRONTEND_URL` | Frontend origin for CORS (e.g., `http://localhost:5173`) |
| `OLLAMA_MODEL` | LLM model name (default: `qwen2`) |
| `CROSS_ENCODER_MODEL` | HuggingFace cross-encoder model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) |

**Frontend** (`frontend/.env`):

| Variable | Description |
|----------|-------------|
| `VITE_API_BASE_URL` | Backend base URL (e.g., `http://localhost:8000/api`) |

**Ingestion** (`ingestion/.env`):

| Variable | Description |
|----------|-------------|
| `UPSTASH_DB_URL` | Upstash Vector REST endpoint |
| `UPSTASH_TOKEN` | Upstash token |
| `UPSTASH_READ_ONLY_TOKEN` | Read-only token |
| `DIMENTION_COUNT` | Vector dimension count (768) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/getResponse` | Main RAG endpoint — FormData: `query`, `image` (file), optional `conversation_id` |
| GET | `/api/conversations` | List all conversations |
| POST | `/api/conversations/create` | Create conversation (`title` in body) |
| GET | `/api/conversations/<id>` | Get conversation with messages |
| DELETE | `/api/conversations/<id>/delete` | Delete conversation |

## Data Ingestion

```bash
# Run from repo root with venv active
python ingestion/generate_medical_embeddings.py   # outputs data/processed/embedding_results.json
python ingestion/upload_vectors.py                # uploads to Upstash
```

Input format (`data/processed/embedding_input.json`):
```json
[{
  "id": "...",
  "text": "...",
  "image": "filename.png",
  "label": "...",
  "cluster": "...",
  "normal": true,
  "cleaned_text": "..."
}]
```

## Utility Notebooks

Run in order from `utils/`. Each notebook produces outputs consumed by the next.

| # | Notebook | Input | Output | Description |
|---|----------|-------|--------|-------------|
| 1 | `remove_unwanted_data.ipynb` | `indiana_reports.csv` | `indiana_reports_with_projections.csv` | Initial quality pass — detects missing/duplicate records, resolves impression vs. findings conflicts, tags normal/abnormal via keyword matching |
| 2 | `link_images_and_report.ipynb` | `indiana_projections.csv` + reports | filtered dataset | Filters to frontal-view images only, deduplicates UIDs, merges projections with reports, removes 162 records with missing frontal images |
| 3 | `IU_data_cleaning_and_classification.ipynb` | merged dataset | `cluster_categorized_data.csv` | Classifies impressions as normal/abnormal using BioGPT; clusters PubMedBERT embeddings into 6 categories via K-Means (Pulmonary Disease, Normal, Mild Negative, Chronic/Mixed, Cardiac/Structural); trains logistic regression classifier (98.6% accuracy) |
| 4 | `preprocess_text.ipynb` | `cluster_categorized_data.csv` | `embedding_input.json` | Strips anonymization tokens, removes measurements, normalizes MeSH terms, assembles final embedding text from impression + findings + indication fields; outputs 3,666 clean records |
| 5 | `knowledge_base_to_vector.ipynb` | `embedding_input.json` + images | `embedding_results.json` | Generates dual 768-dim embeddings per record using Rad-DINO (image `[CLS]` token) and PubMedBERT (mean pooling), both L2-normalized; runs on Google Colab (GPU required) |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Slow startup | Expected — PubMedBERT, Rad-DINO, BM25, and Ollama all load on first request if preloading fails |
| "Ollama model not found" | Run `ollama pull qwen2` and ensure `ollama serve` is running |
| GPU not used | Run `python -c "import torch; print(torch.cuda.is_available())"` and verify `nvidia-smi` is in PATH |
| CORS errors | `FRONTEND_URL` in backend `.env` must match the exact origin (including port) of the frontend |

## Data Source

[Indiana University Chest X-rays — Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
