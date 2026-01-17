from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from transformers import pipeline
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi
import re
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


VECTOR_CACHE = {}


LANGUAGES = [
    'hi', 'en', 'bn', 'te', 'mr',
    'ta', 'ur', 'gu', 'kn', 'ml', 'pa'
]

LANG_MAP = {
    'hi': 'hin_Deva',
    'ur': 'urd_Arab',
    'bn': 'ben_Beng',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'mr': 'mar_Deva',
    'ne': 'npi_Deva',
    'ar': 'arb_Arab'
}

translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M"
)

llm = OllamaLLM(model="llama3.2")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def extract_video_id(url: str):
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)",
        r"youtube\.com\/embed\/([^&\n?#]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def translate_text(text, src_lang):
    try:
        return translator(
            text,
            src_lang=src_lang,
            tgt_lang="eng_Latn"
        )[0]["translation_text"]
    except Exception:
        return text


def fetch_and_translate_transcript(video_id):
    api = YouTubeTranscriptApi()

    transcript_snippets = api.fetch(
        video_id=video_id,
        languages=LANGUAGES
    )

    first_text = " ".join(t.text for t in transcript_snippets[:10])
    detected = detect(first_text)

    if detected != "en" and detected in LANG_MAP:
        translated = [
            translate_text(t.text, LANG_MAP[detected])
            for t in transcript_snippets
        ]
        return " ".join(translated)

    return " ".join(t.text for t in transcript_snippets)


def get_vectorstore(video_id, text):
    if video_id in VECTOR_CACHE:
        return VECTOR_CACHE[video_id]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    store = FAISS.from_texts(chunks, embedding)

    VECTOR_CACHE[video_id] = store
    return store


def build_chain(retriever):
    prompt = PromptTemplate(
        template="""
You are a helpful assistant answering questions from a YouTube video.

Context:
{context}

Question:
{question}

Answer only using the context.
""",
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        RunnableParallel({
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    url = data.get("video_url")
    question = data.get("question")

    if not url or not question:
        return jsonify({"error": "URL and question required"}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    try:
        transcript = fetch_and_translate_transcript(video_id)
        store = get_vectorstore(video_id, transcript)

        retriever = store.as_retriever(search_kwargs={"k": 4})
        chain = build_chain(retriever)

        answer = chain.invoke(question)

        return jsonify({
            "answer": answer,
            "video_id": video_id
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/get_video_info", methods=["POST"])
def get_video_info():
    """Get video info to display preview"""
    data = request.json
    url = data.get("video_url")

    if not url:
        return jsonify({"error": "URL required"}), 400

    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    return jsonify({
        "video_id": video_id,
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
    })


if __name__ == "__main__":
    app.run(debug=True)
