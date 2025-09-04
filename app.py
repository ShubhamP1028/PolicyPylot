#!/usr/bin/env python3
"""
PolicyPilot - AI-powered policy document assistant
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import hashlib
import json

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
CHROMA_DB_PATH = "chroma_db"
CONFIG_FILE = "config.json"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

class APIKeyManager:
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.api_key = self.load_api_key()
    
    def load_api_key(self):
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    return config.get('gemini_api_key', '')
        except:
            pass
        return ''
    
    def save_api_key(self, api_key):
        try:
            config = {'gemini_api_key': api_key}
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
            self.api_key = api_key
            return True
        except:
            return False
    
    def validate_api_key(self, api_key):
        if not api_key or len(api_key) < 20:
            print(f"❌ API key too short: {len(api_key) if api_key else 0} characters")
            return False
        
        # Basic format check for Gemini API key
        if not api_key.startswith('AIza'):
            print(f"❌ API key doesn't start with 'AIza': {api_key[:10]}...")
            return False
        
        # For now, just do basic validation without API call to avoid issues
        print(f"✅ API key format looks valid: {api_key[:10]}...")
        return True
    
    def is_configured(self):
        return bool(self.api_key and len(self.api_key) > 20)

api_manager = APIKeyManager()
GEMINI_API_KEY = api_manager.api_key

class PolicyDocument:
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
    def __init__(self):
        self.embeddings = None
        self.chat_model = None
        self.vectorstore = None
        self.loaded_documents: Dict[str, PolicyDocument] = {}
        self.chat_history: List[Dict] = []
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self._initialize_models()
        self._load_existing_vectorstore()
    
    def _initialize_models(self):
        if api_manager.is_configured():
            try:
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_manager.api_key
                )
                
                self.chat_model = ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    google_api_key=api_manager.api_key
                )
            except Exception as e:
                print(f"⚠️ Error initializing models: {e}")
    
    def update_api_key(self, new_api_key):
        if api_manager.validate_api_key(new_api_key):
            api_manager.save_api_key(new_api_key)
            self._initialize_models()
            return True
        return False
    
    def _load_existing_vectorstore(self):
        if not self.embeddings:
            return
        try:
            if os.path.exists(os.path.join(CHROMA_DB_PATH, "chroma.sqlite3")):
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_DB_PATH,
                    embedding_function=self.embeddings
                )
                print("📚 Loaded existing vector database")
        except Exception as e:
            print(f"⚠️ Could not load existing vectorstore: {e}")
            self.vectorstore = None
    
    def load_document(self, file_path: str, doc_name: str) -> Dict[str, Any]:
        if not self.embeddings:
            return {"success": False, "error": "❌ API key not configured. Please set your Gemini API key first."}
        
        try:
            file_extension = Path(file_path).suffix.lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
                doc_type = "PDF"
            elif file_extension in ['.txt', '.md']:
                loader = TextLoader(file_path, encoding='utf-8')
                doc_type = "Text"
            else:
                return {"success": False, "error": f"Unsupported file type: {file_extension}"}
            
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            policy_doc = PolicyDocument(doc_name, file_path, doc_type)
            policy_doc.chunk_count = len(chunks)
            
            if chunks:
                sample_text = chunks[0].page_content[:500]
                policy_doc.summary = self._generate_summary(sample_text, doc_name)
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source_document': doc_name,
                    'doc_type': doc_type,
                    'chunk_id': f"{policy_doc.id}_{i}",
                    'document_id': policy_doc.id,
                    'page_number': chunk.metadata.get('page', 'Unknown')
                })
            
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=CHROMA_DB_PATH
                )
            else:
                self.vectorstore.add_documents(chunks)
            
            self.vectorstore.persist()
            self.loaded_documents[policy_doc.id] = policy_doc
            
            return {
                "success": True,
                "document": policy_doc.to_dict(),
                "message": f"✅ Successfully loaded {doc_name} with {len(chunks)} chunks"
            }
            
        except Exception as e:
            return {"success": False, "error": f"❌ Error loading document: {str(e)}"}
    
    def _generate_summary(self, text: str, doc_name: str) -> str:
        if not self.chat_model:
            return "Policy document summary unavailable"
        try:
            prompt = f"Briefly summarize this policy document excerpt in 1-2 sentences:\n\nDocument: {doc_name}\nText: {text}\n\nSummary:"
            response = self.chat_model.invoke(prompt)
            return (getattr(response, "content", "") or str(response)).strip()
        except:
            return "Policy document summary unavailable"
    
    def ask_question(self, question: str, active_doc_ids: List[str] = None) -> Dict[str, Any]:
        if not self.chat_model:
            return {
                "answer": "❌ API key not configured. Please set your Gemini API key first.",
                "sources": [],
                "success": False
            }
        
        if self.vectorstore is None:
            return {
                "answer": "Please upload and load policy documents first.",
                "sources": [],
                "success": False
            }
        
        try:
            search_kwargs = {"k": 6}
            if active_doc_ids:
                search_kwargs["filter"] = {"document_id": {"$in": active_doc_ids}}
            
            relevant_docs = self.vectorstore.similarity_search(question, **search_kwargs)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information in the loaded documents.",
                    "sources": [],
                    "success": False
                }
            
            context_text = "\n\n".join([
                f"Source: {doc.metadata.get('source_document', 'Unknown')} (Page {doc.metadata.get('page_number', 'Unknown')})\n{doc.page_content}"
                for doc in relevant_docs
            ])
            
            history_context = ""
            if self.chat_history:
                recent_history = self.chat_history[-3:]
                history_context = "\n".join([
                    f"Previous Q: {item['question']}\nPrevious A: {item['answer'][:200]}..."
                    for item in recent_history
                ])
            
            prompt = f"""You are PolicyPilot, an AI assistant that helps people understand policy documents.

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

Answer:"""
            
            response = self.chat_model.invoke(prompt)
            answer = (getattr(response, "content", "") or str(response)).strip()
            
            sources = []
            for doc in relevant_docs:
                sources.append({
                    "document_name": doc.metadata.get("source_document", "Unknown"),
                    "document_id": doc.metadata.get("document_id", ""),
                    "page_number": doc.metadata.get("page_number", "Unknown"),
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "doc_type": doc.metadata.get("doc_type", "Unknown")
                })
            
            chat_entry = {
                "question": question,
                "answer": answer,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            }
            self.chat_history.append(chat_entry)
            
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
        return [doc.to_dict() for doc in self.loaded_documents.values()]
    
    def toggle_document_active(self, doc_id: str) -> bool:
        if doc_id in self.loaded_documents:
            self.loaded_documents[doc_id].is_active = not self.loaded_documents[doc_id].is_active
            return True
        return False
    
    def get_chat_history(self) -> List[Dict]:
        return self.chat_history
    
    def clear_chat_history(self):
        self.chat_history = []


policy_pilot = GeminiPolicyPilot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        result = policy_pilot.load_document(file_path, filename)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400

@app.route('/api/documents', methods=['GET'])
def get_documents():
    documents = policy_pilot.get_documents()
    return jsonify({"documents": documents})

@app.route('/api/documents/<doc_id>/toggle', methods=['POST'])
def toggle_document(doc_id):
    success = policy_pilot.toggle_document_active(doc_id)
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Document not found"}), 404

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    
    question = data['question']
    active_doc_ids = data.get('active_doc_ids', [])
    
    result = policy_pilot.ask_question(question, active_doc_ids)
    return jsonify(result)

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    history = policy_pilot.get_chat_history()
    return jsonify({"history": history})

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    policy_pilot.clear_chat_history()
    return jsonify({"success": True})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "documents_loaded": len(policy_pilot.loaded_documents),
        "vectorstore_ready": policy_pilot.vectorstore is not None,
        "api_key_configured": api_manager.is_configured()
    })

@app.route('/api/config/status', methods=['GET'])
def get_config_status():
    return jsonify({
        "api_key_configured": api_manager.is_configured(),
        "has_api_key": bool(api_manager.api_key)
    })

@app.route('/api/config/api-key', methods=['POST'])
def set_api_key():
    try:
        data = request.get_json()
        print(f"📥 Received API key request: {data}")
        
        if not data or 'api_key' not in data:
            print("❌ No API key in request data")
            return jsonify({"error": "No API key provided"}), 400
        
        api_key = data['api_key'].strip()
        print(f"🔑 Processing API key: {api_key[:10]}...")
        
        if policy_pilot.update_api_key(api_key):
            print("✅ API key updated successfully")
            return jsonify({
                "success": True,
                "message": "✅ API key configured successfully!"
            })
        else:
            print("❌ API key validation failed")
            return jsonify({
                "success": False,
                "error": "❌ Invalid API key. Please check your key and try again."
            }), 400
    except Exception as e:
        print(f"💥 Error in set_api_key: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)