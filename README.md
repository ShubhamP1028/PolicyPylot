# 🚀 DocuPi - AI-Powered Policy Document Assistant

<div align="center">

![DocuPi Logo](https://img.shields.io/badge/DocuPi-AI%20Document%20Assistant-blue?style=for-the-badge&logo=python)

**Transform your document analysis with intelligent AI-powered conversations**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20Now-green?style=for-the-badge&logo=railway)](https://docupi.up.railway.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

## 📖 Overview

DocuPi is an intelligent document analysis platform that allows you to upload policy documents, legal texts, or any PDF/TXT/Markdown files and have natural conversations with them using AI. Ask questions in plain English and get precise answers with source citations.

### ✨ Key Features

- 🤖 **AI-Powered Q&A** - Natural language conversations with your documents
- 📄 **Multi-Format Support** - PDF, TXT, and Markdown files
- 🔍 **Semantic Search** - Advanced vector-based document retrieval
- 📚 **Source Citations** - Every answer includes page references and document sources
- 🎨 **Beautiful Interface** - Modern, responsive web design
- 🔐 **Secure & Private** - Your documents stay on your system
- ⚡ **Real-time Processing** - Instant document analysis and question answering
- 🌐 **Web-Based** - No installation required, works in any browser

## 🚀 Live Demo

**Try DocuPi right now:** [https://docupi.up.railway.app](https://docupi.up.railway.app)

## 🛠️ Technology Stack

### Backend
- **Flask** - Web framework
- **LangChain** - AI application framework
- **Google Gemini AI** - Large language model
- **ChromaDB** - Vector database for document storage
- **PyPDF** - PDF processing

### Frontend
- **HTML5/CSS3** - Modern web interface
- **JavaScript** - Interactive functionality
- **Font Awesome** - Icons and UI elements

### Deployment
- **Railway** - Cloud hosting platform
- **Python 3.8+** - Runtime environment

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key ([Get yours here](https://makersuite.google.com/app/apikey))
- Modern web browser

## 🚀 Quick Start

### Option 1: Use the Live Demo
1. Visit [https://docupi.up.railway.app](https://docupi.up.railway.app)
2. Enter your Google Gemini API key when prompted
3. Upload a document and start asking questions!

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/shubhamp1028/PolicyPylot.git
   cd PolicyPylot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5050`

## 📁 Project Structure

```
PolicyPylot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Railway deployment configuration
├── templates/
│   └── index.html        # Web interface
├── uploads/              # Document storage (created automatically)
├── chroma_db/           # Vector database (created automatically)
└── config.json          # API key storage (created automatically)
```

## 🔧 Configuration

### Environment Variables
- `PORT` - Server port (automatically set by Railway)
- `GEMINI_API_KEY` - Your Google Gemini API key (set via web interface)

### API Key Setup
1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Enter it in the web interface when prompted
3. The key is securely stored locally

## 📖 How to Use

1. **Upload Documents**
   - Click "Add Document" to upload PDF, TXT, or MD files
   - Documents are automatically processed and indexed

2. **Ask Questions**
   - Type your question in natural language
   - Get instant answers with source citations
   - View relevant document sections

3. **Manage Documents**
   - Toggle documents on/off for search
   - View document summaries and metadata
   - Clear chat history when needed

## 🎯 Use Cases

- **Legal Professionals** - Analyze contracts and legal documents
- **HR Teams** - Review company policies and procedures
- **Researchers** - Study complex academic papers
- **Students** - Understand lengthy textbooks and articles
- **Business Analysts** - Extract insights from reports
- **Anyone** - Who needs to quickly understand complex documents

## 🔒 Privacy & Security

- **Local Processing** - Documents are processed on your system
- **Secure Storage** - API keys are encrypted and stored locally
- **No Data Sharing** - Your documents never leave your control
- **Temporary Storage** - Uploaded files are stored temporarily for processing

## 🚀 Deployment

### Railway (Recommended)
1. Fork this repository
2. Connect your GitHub account to [Railway](https://railway.app)
3. Deploy from GitHub repository
4. Railway will automatically detect and deploy your Flask app

### Other Platforms
- **Render** - Connect GitHub repo and deploy
- **Heroku** - Use the included Procfile
- **DigitalOcean** - App Platform deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Shubham Pandey**
- GitHub: [@yourusername](https://github.com/shubhamp1028)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/shubham1028)
- Email: your.email@example.com

## 🙏 Acknowledgments

- [LangChain](https://langchain.com) for the AI framework
- [Google Gemini](https://ai.google.dev) for the language model
- [ChromaDB](https://www.trychroma.com) for vector storage
- [Railway](https://railway.app) for hosting platform

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/PolicyPylot?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/PolicyPylot?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/PolicyPylot)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/PolicyPylot)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by [Shubham Pandey](https://github.com/shubhamp1028)

</div>