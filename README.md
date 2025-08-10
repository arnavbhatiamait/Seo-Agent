# Seo-Agent

An AI-powered YouTube SEO assistant that helps optimize videos by generating high-quality titles, descriptions, tags/keywords, and thumbnail concepts — with multilingual support and plug-and-play compatibility with multiple LLMs. Built with Streamlit for a fast, simple, and interactive UI.

- Multi-LLM: OpenAI, Gemini, Grok, Ollama (local) supported.
- Multilingual generation: Suggests SEO assets in multiple languages for global reach.
- YouTube-focused: Optimizes title, description, tags, outline, and thumbnail brief.
- Streamlit app: One-command launch and browser UI.
- Analysis helpers: Utilities for content scoring and suggestions (utils/analysis_functions.py).
![pic1](https://github.com/arnavbhatiamait/Seo-Agent/blob/main/Screenshot%202025-08-10%20221528.png)
![pic2](https://github.com/arnavbhatiamait/Seo-Agent/blob/main/Screenshot%202025-08-10%20221814.png)
![pic3](https://github.com/arnavbhatiamait/Seo-Agent/blob/main/Screenshot%202025-08-10%20221908.png)
![pic4](https://github.com/arnavbhatiamait/Seo-Agent/blob/main/Screenshot%202025-08-10%20221915.png)
![pic5](https://github.com/arnavbhatiamait/Seo-Agent/blob/main/Screenshot%202025-08-10%20221925.png)
![pic6](https://github.com/arnavbhatiamait/Seo-Agent/blob/main/Screenshot%202025-08-10%20221935.png)
![pic7](https://github.com/arnavbhatiamait/Seo-Agent/blob/main/Screenshot%202025-08-10%20221950.png)
![pic8](https://github.com/arnavbhatiamait/Seo-Agent/blob/main/Screenshot%202025-08-10%20221957.png)



## Features

- Title generator: Click-worthy, keyword-rich YouTube titles aligned to intent.
- Description writer: Long-form, SEO-optimized descriptions with CTAs and hashtags.
- Tag/keyword suggestions: Relevant, high-intent tags to improve discoverability.
- Thumbnail brief creator: Visual hooks, text overlays, and contrast suggestions for higher CTR.
- Multilingual outputs: Generate all assets in multiple languages with a single toggle.
- Model flexibility:
  - OpenAI (e.g., GPT-4 class models)
  - Google Gemini
  - Groq
  - Ollama for local models (privacy-friendly, no external calls)
- Streamlit UI: Simple, fast, deploy anywhere (local, Streamlit Cloud, VM).

## Demo

- Launch the app and experiment with YouTube keywords, topics, or drafts to generate SEO assets.
- Demo screenshots are included in the repository for a quick preview of the UI and outputs.



## Project Structure

- app.py — Streamlit app entrypoint with UI and LLM orchestration.
- app.ipynb — Notebook for experiments/workflows.
- utils/ — Helper functions and analysis utilities.
- text/ — Example content or prompts (if included).
- images — Demo screenshots (root).

## Getting Started

### Prerequisites

- Python 3.10+ recommended.
- Streamlit installed.
- At least one LLM provider configured:
  - OpenAI (OPENAI_API_KEY)
  - Google Gemini (GEMINI_API_KEY)
  - Groq
  - Ollama running locally (ollama serve)

Note on Streamlit SEO: Streamlit apps are client-rendered; metadata customization may require workarounds if aiming to index the app itself, not its outputs.

### Installation

1) Clone the repository:
- git clone arnavbhatiamait/Seo-Agent
- cd Seo-Agent

2) Create and activate a virtual environment:
- python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)

3) Install dependencies:
- pip install -r requirements.txt  (or pip install streamlit openai google-generativeai)

4) Set environment variables (choose providers you’ll use):
- export OPENAI_API_KEY=...
- export GEMINI_API_KEY=...
- export Groq_API_KEY=...  (Groq)
- Ensure Ollama is running for local models: ollama serve

5) Run the app:
- streamlit run app.py

## How It Works

- Input: Provide a topic, seed keyword(s), target audience, and language preference in the UI.
- Model selection: Choose OpenAI/Gemini/Grok or local Ollama for generation.
- Generation pipeline:
  - Title variants → pick best CTR option.
  - Description with keyword placement, CTAs, timestamps outline (optional).
  - Tags/keywords list with search-intent hints.
  - Thumbnail concept brief with hook, colors, and overlay text.
- Output: Copy-ready assets with multilingual support.

## Configuration

- Model and provider can be selected from the sidebar/settings panel in the Streamlit UI.
- Language setting applies across title, description, tags, and thumbnail brief.
- Optional: Integrate external SEO APIs for volume/competition checks if extending the app.

## Extending

- Add a competitor/title-analyzer step to benchmark CTR hooks.
- Integrate RapidAPI SEO analyzers for page checks or channel audits.
- Add automatic A/B title generation and scoring using utility hooks in utils/.
- Plug in YouTube Data API for channel-based keyword discovery and trend mapping.

## Deployment

- Local: streamlit run app.py.
- Cloud: Streamlit Community Cloud, Render, or Heroku; set env vars in the platform dashboard.
- For Ollama-only setups, ensure the server is accessible on the deploy target.

## Roadmap

- One-click asset export (CSV/JSON).
- Batch generation for content calendars.
- SERP/competitor scraping for data-enhanced prompts.
- Basic thumbnail template exports.

## Acknowledgements

- Built with Streamlit for the app layer.
- Inspired by agentic content-generation workflows commonly used for SEO and YouTube optimization.



.
