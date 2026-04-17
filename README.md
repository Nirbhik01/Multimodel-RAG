# Multimodel-RAG
AI-powered multimodal RAG application that combines chest X-ray images and clinical symptoms to retrieve similar cases and provide explainable insights.

# Project Structure

```text
root/
├── .venv/
├── backend/
├── chest-x-ray-data/
│   ├── images/
│   ├── indiana_projections.csv
│   └── indiana_reports.csv
├── frontend/
├── rag_pipeline/
├── utils/
│   └── link_images_and_report.ipynb
|   └── preprocess_text.ipynb
|   └── remove_unwanted_data.ipynb
├── .gitignore
└── README.md
```

# Project Flow
- Utilities
    1. remove_unwanted_data.ipynb
    2. link_images_and_report.ipynb
    3. preprocess_text.ipynb

# Data
https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university