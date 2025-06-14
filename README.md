# Virtual Teaching Assistant — TDS Project 1 (IITM BSc)

This project implements a Virtual Teaching Assistant for the Tools in Data Science (TDS) course in IIT Madras Online BSc Program (Jan 2025).

It builds an intelligent question answering system by scraping course lecture notes and Discourse forum posts, generating embeddings, and serving queries via an API.

---

## 🔧 Features

- ✅ Scrapes Discourse posts (1 Jan 2025 to 14 Apr 2025)
- ✅ Scrapes TDS Markdown lecture content (as on 15 Apr 2025)
- ✅ Uses OpenAI-compatible embeddings via AIPipe (text-embedding-3-small)
- ✅ Full semantic search using SQLite
- ✅ FastAPI backend for serving queries
- ✅ Optional multimodal queries (image+text)
- ✅ Deployable on Vercel or any platform

---

## 📂 Project Structure

| File | Description |
| ---- | ----------- |
| `Scraper.py` | Discourse scraper |
| `Crawler.py` | Markdown lecture notes crawler |
| `preprocess.py` | Preprocesses and generates embeddings |
| `app.py` | FastAPI backend to serve queries |
| `requirements.txt` | Dependencies |
| `vercel.json` | Deployment config for Vercel |
| `.env` | Stores API keys (local only, not uploaded) |
| `downloaded_threads/` | Scraped Discourse data |
| `markdown_files/` | Scraped Markdown lecture notes |
| `knowledge_base.db` | SQLite knowledge base |

---

## 🚀 Setup Instructions

### ✅ 1. Clone Repository

```bash
git clone https://github.com/VedantKakde/virtual-ta-tds-project.git
cd virtual-ta-tds-project
✅ 2. Create Python Environment
bash
Copy
Edit
conda create -n virtual-ta python=3.10
conda activate virtual-ta
pip install -r requirements.txt
✅ 3. Install Playwright Browsers
bash
Copy
Edit
python -m playwright install
✅ 4. Configure AIPipe API Key
Create .env file:

bash
Copy
Edit
API_KEY=your_aipipe_api_key_here
Sign up for free API key at AIPipe.org

✅ 5. Run Scrapers
Discourse
bash
Copy
Edit
python Scraper.py
Markdown
bash
Copy
Edit
python Crawler.py
✅ 6. Preprocessing & Embedding
bash
Copy
Edit
python preprocess.py
✅ 7. Run Backend API
bash
Copy
Edit
uvicorn app:app --reload
Open docs at:
http://127.0.0.1:8000/docs

📄 Submission Info (for IITM TDS Project 1)
Fully compliant with project instructions.

Exposes API that accepts:

json
Copy
Edit
{
  "question": "...",
  "image": null
}
Returns answer + sources as required.

All code and scraper scripts included as bonus.

📝 License
This project is licensed under the MIT License.

✅ This repo is fully functional and ready for evaluation.
