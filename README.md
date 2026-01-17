# 🎥 YouTube AI Chatbot

An intelligent chatbot web application that allows you to ask questions about any YouTube video. Powered by AI, it fetches video transcripts, translates them if needed, and answers your questions using advanced language models.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎯 **Ask Questions About Videos** - Paste any YouTube URL and ask questions about the content
- 💬 **Chat History** - Have multi-turn conversations about the same video
- 🖼️ **Video Preview** - See video thumbnail and title before asking questions
- 🌍 **Multi-Language Support** - Automatically detects and translates non-English videos
- 🤖 **AI-Powered** - Uses Ollama (LLaMA 3.2) and RAG for accurate answers
- 🎨 **Modern UI** - Beautiful glassmorphism design with smooth animations
- ⚡ **Fast & Efficient** - Vector caching for quick follow-up questions
- 📱 **Responsive** - Works perfectly on desktop and mobile devices

## 🛠️ Technologies Used

- **Backend**: Flask (Python)
- **AI/ML**: 
  - LangChain for orchestration
  - Ollama (LLaMA 3.2) for question answering
  - HuggingFace Embeddings (all-MiniLM-L6-v2)
  - Facebook NLLB for translation
- **Vector Store**: FAISS
- **Frontend**: HTML, CSS, Vanilla JavaScript
- **APIs**: 
  - YouTube Transcript API
  - YouTube oEmbed API

## 📋 Prerequisites

Before running this application, make sure you have:

1. **Python 3.10+** installed
2. **Ollama** installed and running with LLaMA 3.2 model
   ```bash
   # Install Ollama from https://ollama.ai
   # Then pull the model:
   ollama pull llama3.2
   ```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Youtube_Chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv310
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv310\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source venv310/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

## 🏃 Running the Application

1. **Make sure Ollama is running**
   ```bash
   ollama serve
   ```

2. **Start the Flask application**
   ```bash
   python app.py
   ```

3. **Open your browser**
   ```
   http://localhost:5000
   ```

## 💡 How to Use

1. **Enter YouTube URL**: Paste any YouTube video URL in the input field
2. **See Preview**: The video thumbnail and title will appear automatically
3. **Ask Your Question**: Type your question about the video content
4. **Get AI Answer**: Receive an intelligent answer based on the video transcript
5. **Continue Chatting**: Ask follow-up questions without re-entering the URL
6. **New Video**: Click "New Video" button to start analyzing a different video

## 📁 Project Structure

```
Youtube_Chatbot/
│
├── app.py                  # Flask backend application
├── Chatbot.ipynb          # Jupyter notebook (development/testing)
├── requirement.txt        # Python dependencies
├── README.md              # Project documentation
│
├── templates/
│   └── index.html         # Frontend HTML with CSS and JavaScript
│
├── venv310/               # Virtual environment (not in git)
└── __pycache__/           # Python cache files (not in git)
```

## 🔧 Configuration

### Changing the AI Model

Edit `app.py` and modify:
```python
llm = OllamaLLM(model="llama3.2")  # Change to any Ollama model
```

### Adjusting Chunk Size

Modify the text splitter settings:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Adjust this
    chunk_overlap=100    # Adjust this
)
```

### Changing Vector Store

Currently uses FAISS. To change to another vector store, modify the `get_vectorstore` function.

## 🎯 Key Features Explained

### 1. RAG (Retrieval Augmented Generation)
The app uses RAG to provide accurate answers by:
- Splitting video transcripts into chunks
- Creating vector embeddings
- Retrieving relevant chunks for each question
- Passing context to the LLM

### 2. Vector Caching
Videos are processed once and cached, making follow-up questions instant.

### 3. Language Detection & Translation
Automatically detects video language and translates to English using Facebook's NLLB model.

### 4. Multi-Turn Conversations
Chat interface maintains context for natural conversations about the same video.

## 🐛 Troubleshooting

### Ollama Connection Error
```
Make sure Ollama is running: ollama serve
Check if llama3.2 model is installed: ollama list
```

### Module Not Found Error
```bash
pip install -r requirement.txt
```

### Video Transcript Not Available
Some videos don't have transcripts. Try another video with captions enabled.

### Slow Response Time
First question takes longer (loading models + processing transcript). Follow-up questions are faster due to caching.

## 🚀 Future Enhancements

- [ ] Add video timestamp citations in answers
- [ ] Support for video playlists
- [ ] Export Q&A as PDF
- [ ] Support for more languages
- [ ] User authentication and saved conversations
- [ ] Video summarization feature
- [ ] Dark mode toggle

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing framework
- [Ollama](https://ollama.ai) for local LLM inference
- [HuggingFace](https://huggingface.co) for embeddings and translation models
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript extraction

---

⭐ If you found this project helpful, please give it a star!

## 📞 Support

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]
