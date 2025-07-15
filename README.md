# Reddit Persona Analyzer

A powerful and intelligent Streamlit web app for analyzing Reddit users or subreddit communities. The app scrapes Reddit content and uses **Gemini 1.5** to generate deep-dive persona reports or community overviews, with full **citation tracking** and **export support**.

---

## 🚀 Live Demo

👉 [Launch the app on Streamlit](https://reddit-persona-analyzer-agent.streamlit.app/)

---

## 🔍 Features

### For Reddit Users

* Scrapes recent posts & comments
* Cleans and preprocesses content
* Generates **detailed persona profiles** (e.g., age, tone, interests)
* Each insight includes **citations** to specific post/comment IDs

### For Subreddits

* Identifies top contributors
* Summarizes key discussion themes
* Analyzes community culture & dynamics
* Offers actionable insights for engagement

### Extras

* One-click download of generated reports (with citation reference)
* Error-handled API retries with exponential backoff
* Cloud-ready: compatible with **Streamlit Secrets**

---

## 🧠 How It Works

1. **Input**: Reddit profile/subreddit URL and Gemini API Key
2. **Scrape**: Collects data using `praw` (Reddit API)
3. **Analyze**: Gemini AI creates structured insights
4. **Output**: Persona/Community report with detailed citations

---

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: Python 3, PRAW (Reddit API), Google Generative AI
* **AI Model**: Gemini 1.5 Flash
* **Export**: Downloads via Streamlit buttons

---

## 🔐 Setup Instructions

### 1. Clone the Repo

```bash
https://github.com/Balaji1472/Reddit-Persona-Analyzer-Agent.git
cd Reddit-Persona-Analyzer-Agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Use **Streamlit Secrets** or a `.env` file:

```env
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_app_name
```

You will also need a **Gemini API Key** from Google.

### 4. Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
reddit-persona-analyzer/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Dependencies
├── .env or .streamlit/secrets.toml  # API credentials
```

---

## 📸 Screenshots

> *Add screenshots here showing Persona report, Subreddit insights, and download button.*

---

## 💡 Use Cases

* Competitive subreddit research
* Influencer profiling
* Market research
* Reddit user behavior studies

---

## 📬 Contact

**Author**: Balaji V

**Email**: [balajirama.2005@gmail.com](mailto:balajirama.2005@gmail.com)

**GitHub**: [Balaji1472/](https://github.com/Balaji1472/)

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for more information.

---

## 🌐 Acknowledgments

* Google for AI APIs
* Reddit for the content & API
* Streamlit for interactive UI

---

> Made with ❤️ for AI-powered digital research
