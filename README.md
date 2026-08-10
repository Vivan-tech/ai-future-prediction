# AI Future Prediction Dashboard

A Streamlit app that lets you upload any dataset, automatically picks a
suitable machine learning model (classification or regression), shows
predictions and graphs, lets you try live "what-if" predictions, and
generates a downloadable PDF report.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL it prints (usually `http://localhost:8501`).

## Get a live public link (free, ~5 minutes) — Streamlit Community Cloud

This is the simplest way to get a link you can send to anyone.

1. **Create a free GitHub account** (if you don't have one): https://github.com/signup
2. **Create a new repository** (e.g. `ai-future-prediction`), and upload
   these three files to it: `app.py`, `requirements.txt`, `README.md`.
   - Easiest way: on the repo page click **"Add file" → "Upload files"**
     and drag the files in, then click **Commit changes**.
3. Go to **https://share.streamlit.io** and sign in with your GitHub account.
4. Click **"New app"**, choose your repository, branch `main`, and set the
   main file path to `app.py`.
5. Click **Deploy**. In about a minute it will give you a public URL like:
   `https://your-app-name.streamlit.app`
6. Share that link with anyone — it opens the live dashboard in their
   browser, no install needed.

That's it — any time you push new changes to the GitHub repo, the live
app updates automatically.
