# 📄 PDF RAG Chatbot

An AI-powered chatbot that enables natural language conversations with PDF documents using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 📄 **PDF Upload** - Upload and process any PDF document
- 💬 **Interactive Chat** - Ask questions in natural language
- 🎨 **Multiple Answer Styles** - Choose detailed, concise, or bullet-point answers
- 📚 **Source Citations** - See which document sections were used
- 🔍 **Semantic Search** - Find relevant information without exact keywords
- 🚀 **Fast & Accurate** - Powered by Google Gemini 2.5 Flash

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM:** Google Gemini 2.5 Flash
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector Database:** ChromaDB
- **PDF Processing:** pypdf
- **Framework:** LangChain

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_api_key_here
```

5. **Run the application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

1. **Upload PDF** - Click the file uploader in the sidebar
2. **Wait for Processing** - Takes 10-30 seconds depending on document size
3. **Ask Questions** - Type your questions in the chat interface
4. **Get Answers** - Receive AI-generated answers with source citations

## 💡 Example Questions

- "What is this document about?"
- "Summarize the main points in bullet points"
- "What are the key findings?"
- "Explain [specific topic] from the document"
- "What methods are discussed?"

## 🏗️ Architecture
```
PDF Upload
    ↓
Text Extraction & Cleaning
    ↓
Text Chunking (500 chars, 100 overlap)
    ↓
Embedding Generation (384-dim vectors)
    ↓
Vector Database Storage (ChromaDB)
    ↓
User Question → Semantic Search
    ↓
Relevant Chunks Retrieved
    ↓
LLM (Gemini) + Context → Natural Answer
```

## 📁 Project Structure
```
pdf-rag-chatbot/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in git)
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
└── data/                    # Sample PDFs (optional)
```

## 🎯 Features Breakdown

### Answer Styles

- **Detailed** - Comprehensive answers with examples
- **Concise** - Brief 2-3 sentence responses
- **Bullets** - Organized bullet-point format

### Source Citations

Every answer includes references to the specific document chunks used, allowing users to verify information and explore further.

## 🔧 Configuration

You can modify these settings in `app.py`:
```python
# Chunking parameters
chunk_size = 500
chunk_overlap = 100

# Number of chunks to retrieve
n_results = 3

# LLM temperature
temperature = 0
```

## 🚀 Deployment

### Deploy to Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Add `GOOGLE_API_KEY` to Secrets
5. Deploy!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built as part of an 8-week AI learning sprint
- Powered by Google Gemini API
- Vector search with ChromaDB
- Embeddings by sentence-transformers

## 📧 Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter)

Project Link: [https://github.com/YOUR_USERNAME/pdf-rag-chatbot](https://github.com/YOUR_USERNAME/pdf-rag-chatbot)

## ⭐ Show your support

Give a ⭐️ if this project helped you!