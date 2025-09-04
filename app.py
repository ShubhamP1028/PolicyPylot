#!/usr/bin/env python3
"""
PolicyPilot Flask Backend
Modern Flask API with Google Gemini integration and ChromaDB vector store
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import hashlib

# Document processing
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# No external safety types import; use model defaults

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Configuration
GEMINI_API_KEY = "AIzaSyDVUKVdaq6WIYO4zEjFW2ZXlNCAad6RwbY"
UPLOAD_FOLDER = "uploads"
CHROMA_DB_PATH = "chroma_db"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# No direct Google Generative AI client configuration needed; API key passed to clients

class PolicyDocument:
    """Represents a policy document with metadata"""
    
    def __init__(self, name: str, file_path: str, doc_type: str, upload_date: str = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.file_path = file_path
        self.doc_type = doc_type
        self.upload_date = upload_date or datetime.now().isoformat()
        self.chunk_count = 0
        self.is_active = True
        self.file_hash = self._calculate_hash()
        self.summary = ""
    
    def _calculate_hash(self) -> str:
        """Calculate file hash for change detection"""
        try:
            with open(self.file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'doc_type': self.doc_type,
            'upload_date': self.upload_date,
            'chunk_count': self.chunk_count,
            'is_active': self.is_active,
            'summary': self.summary
        }


class GeminiPolicyPilot:
    """Main PolicyPilot class with Google Gemini integration"""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        
        # Initialize Gemini chat model via LangChain integration
        self.chat_model = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GEMINI_API_KEY
        )
        
        # Initialize components
        self.vectorstore = None
        self.loaded_documents: Dict[str, PolicyDocument] = {}
        self.chat_history: List[Dict] = []
        
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Try to load existing vectorstore
        self._load_existing_vectorstore()
    
    def _load_existing_vectorstore(self):
        """Load existing ChromaDB if it exists"""
        try:
            if os.path.exists(os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")):
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=self.embeddings
                )
                print("Loaded existing vector database")
        except Exception as e:
            print(f"Could not load existing vectorstore: {e}")
            self.vectorstore = None
    
    def load_document(self, file_path: str, doc_name: str) -> Dict[str, Any]:
        """Load and process a policy document"""
        try:
            # Determine document type and loader
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                doc_type = "PDF"
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
                doc_type = "Text"
            else:
                return {"success": False, "error": f"Unsupported file type: {file_extension}"}
            
            # Load and split document
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            # Create policy document
            policy_doc = PolicyDocument(doc_name, file_path, doc_type)
            policy_doc.chunk_count = len(chunks)
            
            # Generate summary
            if chunks:
                sample_text = chunks[0].page_content[:500]
                policy_doc.summary = self._generate_summary(sample_text, doc_name)
            
            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source_document': doc_name,
                    'doc_type': doc_type,
                    'chunk_id': f"{policy_doc.id}_{i}",
                    'document_id': policy_doc.id,
                    'page_number': chunk.metadata.get('page', 'Unknown')
                })
            
            # Initialize or update vectorstore
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=CHROMA_DB_PATH
                )
            else:
                self.vectorstore.add_documents(chunks)
            
            # Save vectorstore
            self.vectorstore.persist()
            
            # Store document
            self.loaded_documents[policy_doc.id] = policy_doc
            
            return {
                "success": True,
                "document": policy_doc.to_dict(),
                "message": f"Successfully loaded {doc_name} with {len(chunks)} chunks"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Error loading document: {str(e)}"}
    
    def _generate_summary(self, text: str, doc_name: str) -> str:
        """Generate a brief summary of the document"""
        try:
            prompt = f"""
            Briefly summarize this policy document excerpt in 1-2 sentences:
            
            Document: {doc_name}
            Text: {text}
            
            Summary:
            """
            response = self.chat_model.invoke(prompt)
            return (getattr(response, "content", "") or str(response)).strip()
        except:
            return "Policy document summary unavailable"
    
    def ask_question(self, question: str, active_doc_ids: List[str] = None) -> Dict[str, Any]:
        """Process a user question and return answer with sources"""
        if self.vectorstore is None:
            return {
                "answer": "Please upload and load policy documents first.",
                "sources": [],
                "success": False
            }
        
        try:
            # Filter by active documents if specified
            search_kwargs = {"k": 6}
            if active_doc_ids:
                search_kwargs["filter"] = {"document_id": {"$in": active_doc_ids}}
            
            # Retrieve relevant documents
            relevant_docs = self.vectorstore.similarity_search(question, **search_kwargs)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the loaded documents.",
                    "sources": [],
                    "success": False
                }
            
            # Prepare context for Gemini
            context_text = "\n\n".join([
                f"Source: {doc.metadata.get('source_document', 'Unknown')} (Page {doc.metadata.get('page_number', 'Unknown')})\n{doc.page_content}"
                for doc in relevant_docs
            ])
            
            # Create chat history context
            history_context = ""
            if self.chat_history:
                recent_history = self.chat_history[-3:]  # Last 3 exchanges
                history_context = "\n".join([
                    f"Previous Q: {item['question']}\nPrevious A: {item['answer'][:200]}..."
                    for item in recent_history
                ])
            
            # Generate response using Gemini
            prompt = f"""
            You are PolicyPilot, an AI assistant that helps people understand policy documents.
            
            Previous conversation context:
            {history_context}
            
            Current question: {question}
            
            Relevant policy information:
            {context_text}
            
            Instructions:
            1. Answer the question based ONLY on the provided policy information
            2. Be clear, concise, and authoritative
            3. If the information is not in the provided context, say so
            4. Use bullet points or numbered lists when appropriate
            5. Reference specific sections or pages when possible
            6. Be helpful but precise
            
            Answer:
            """
            
            response = self.chat_model.invoke(prompt)
            answer = (getattr(response, "content", "") or str(response)).strip()
            
            # Format sources
            sources = []
            for doc in relevant_docs:
                sources.append({
                    "document_name": doc.metadata.get("source_document", "Unknown"),
                    "document_id": doc.metadata.get("document_id", ""),
                    "page_number": doc.metadata.get("page_number", "Unknown"),
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "doc_type": doc.metadata.get("doc_type", "Unknown")
                })
            
            # Store in chat history
            chat_entry = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
            self.chat_history.append(chat_entry)
            
            # Keep only last 10 exchanges
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def get_documents(self) -> List[Dict]:
        """Get all loaded documents"""
        return [doc.to_dict() for doc in self.loaded_documents.values()]
    
    def toggle_document_active(self, doc_id: str) -> bool:
        """Toggle document active status"""
        if doc_id in self.loaded_documents:
            self.loaded_documents[doc_id].is_active = not self.loaded_documents[doc_id].is_active
            return True
        return False
    
    def get_chat_history(self) -> List[Dict]:
        """Get chat history"""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history = []


# Initialize PolicyPilot instance
policy_pilot = GeminiPolicyPilot()


# Routes
@app.route('/')
def index():
    """Serve the main application"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        # Save file
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process document
        result = policy_pilot.load_document(file_path, filename)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get all loaded documents"""
    documents = policy_pilot.get_documents()
    return jsonify({"documents": documents})

@app.route('/api/documents/<doc_id>/toggle', methods=['POST'])
def toggle_document(doc_id):
    """Toggle document active status"""
    success = policy_pilot.toggle_document_active(doc_id)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Document not found"}), 404

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Process a question"""
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data['question']
    active_doc_ids = data.get('active_doc_ids', [])
    
    result = policy_pilot.ask_question(question, active_doc_ids)
    return jsonify(result)

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history"""
    history = policy_pilot.get_chat_history()
    return jsonify({"history": history})

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    policy_pilot.clear_chat_history()
    return jsonify({"success": True})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "documents_loaded": len(policy_pilot.loaded_documents),
        "vectorstore_ready": policy_pilot.vectorstore is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)