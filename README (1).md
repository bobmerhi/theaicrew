# AI Crew

A Streamlit portal that connects NVIDIA NIM, Groq, OpenRouter and Google AI Studio, and lets
you run a crew of AI agents (Manager, Researcher, Coder, Builder, Reviewer, plus any roles you add).

## Features
- **Chat** with any single agent, streaming replies, per-agent history.
- **Crew run**: Manager plans, agents work in order seeing earlier outputs, Manager writes the final summary. Download the result as Markdown.
- **Manage crew**: edit name, provider, model, temperature and system prompt; add or delete positions; export/import the crew as JSON.
- Every agent can use a different provider. One click puts the whole crew on a single provider.

## Run locally
```bash
pip install -r requirements.txt
cp secrets.toml.example .streamlit/secrets.toml   # then fill in your keys
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt` and `README.md` to the repo (not the secrets file).
2. In Streamlit Cloud, set the main file to `app.py`.
3. App -> Settings -> Secrets: paste the contents of `secrets.toml.example` with your real keys.

Keys can also be pasted in the sidebar; those last only for your browser session.
You only need keys for the providers you actually use.

## Notes
- All four providers are called through their OpenAI-compatible endpoints, so one client handles them.
- Model IDs change over time. If a call fails with "model not found", edit the Model ID in **Manage crew**.
- Free OpenRouter models are rate-limited and sometimes unavailable.
