from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import PyPDF2
import io
import uuid
from typing import List, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel
import logging

from dotenv import load_dotenv
import os

load_dotenv()

COLLECTION_NAME = "documents"
QDRANT_HOST = os.getenv("QDRANT_HOST")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Document Q&A", description="Upload PDFs and ask questions")
templates = Jinja2Templates(directory=".")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients and models
qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=OPENAI_API_KEY)


class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None

class DocumentChunk:
    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata

def initialize_qdrant():
    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection {COLLECTION_NAME} already exists")
    except Exception:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384,distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=2)
        )
        print(f"Created new collection: {COLLECTION_NAME}")

def extract_text_from_pdf(pdf_file: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks

def embed_text(text: str) -> List[float]:
    try:
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

def store_document_chunks(chunks: List[str], document_id: str, filename: str):
    points = []

    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk,
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i
            }
        )
        points.append(point)

    try:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        logger.info(f"Stored {len(points)} chunks for document {document_id}")
    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        raise

def search_similar_chunks(query: str, document_id: Optional[str] = None, top_k: int = 5) -> List[dict]:
    try:
        query_embedding = embed_text(query)

        search_filter = None
        if document_id:
            search_filter = {
                "must": [
                    {"key": "document_id", "match": {"value": document_id}}
                ]
            }

        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            query_filter=search_filter,
            limit=top_k
        )

        return [
            {
                "text": result.payload["text"],
                "score": result.score,
                "filename": result.payload["filename"],
                "document_id": result.payload["document_id"]
            }
            for result in search_results
        ]
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        raise

def generate_answer(question: str, context_chunks: List[dict]) -> str:
    try:
        context = "\n\n".join([chunk["text"] for chunk in context_chunks])

        prompt = f"""
        Based on the following context from the document(s), answer the question. 
        If the answer cannot be found in the context, say "I cannot find the answer in the provided document(s)."
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.25)

        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        # Fallback to simple context-based response
        return f"Based on the document content, here are the most relevant sections:\n\n" + \
               "\n\n".join([chunk["text"][:200] + "..." for chunk in context_chunks[:2]])

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.on_event("startup")
async def startup_event():
    initialize_qdrant()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Read PDF content
        pdf_content = await file.read()

        # Extract text from PDF
        text = extract_text_from_pdf(pdf_content)

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text found in PDF")

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Chunk the text
        chunks = chunk_text(text)

        # Store chunks in Qdrant
        store_document_chunks(chunks, document_id, file.filename)

        return {
            "message": "PDF uploaded and processed successfully",
            "document_id": document_id,
            "filename": file.filename,
            "chunks_count": len(chunks)
        }

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Search for relevant chunks
        relevant_chunks = search_similar_chunks(
            request.question, 
            request.document_id,
            top_k=5
        )

        # print("RELEVANT CHUNKS: ", relevant_chunks)

        if not relevant_chunks:
            return {
                "answer": "No relevant information found in the uploaded document(s).",
                "sources": []
            }

        # Generate answer using the chunks
        answer = generate_answer(request.question, relevant_chunks)

        return {
            "answer": answer,
            "sources": relevant_chunks,
            "question": request.question
        }

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        # Get all points from the collection
        points = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            with_payload=True
        )[0]

        # Group by document_id
        documents = {}
        for point in points:
            doc_id = point.payload["document_id"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "filename": point.payload["filename"],
                    "chunks_count": 0
                }
            documents[doc_id]["chunks_count"] += 1

        return {"documents": list(documents.values())}

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks"""
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector={
                "filter": {
                    "must": [
                        {"key": "document_id", "match": {"value": document_id}}
                    ]
                }
            }
        )

        return {"message": f"Document {document_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)