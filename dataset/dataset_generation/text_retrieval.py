import os
import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import glob
import argparse
from openai import OpenAI
from typing import List, Dict, Tuple, Any, Optional

this_dir = os.path.dirname(__file__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TextChunk:
    def __init__(self, text):
        self.text = text
        self.embedding = None
    
    def __repr__(self):
        return self.text

class TextParser:
    def parse_file(self, filepath):
        chunks = []

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            # Split content into paragraphs based on double newlines
            paragraphs = content.split('\n\n')
            
            # Create TextChunk objects for non-empty paragraphs
            for para in paragraphs:
                para = para.strip()
                if para:  # Only add non-empty paragraphs
                    chunks.append(TextChunk(para))
                    
        except (UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Error parsing {filepath}: {str(e)}")
        
        print(f"Found {len(chunks)} text chunks")
        return chunks


class EmbeddingEngine:
    """Engine to compute and compare embeddings for code chunks."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize with a sentence transformer model."""
        self.model_name = model_name
        self.total_tokens = 0
        
    def compute_embedding(self, text) -> np.ndarray:
        """Compute embedding for a single code snippet."""
        try:
            response = client.embeddings.create(input=text, model=self.model_name)
        except Exception as e:
            print(e)
            response = client.embeddings.create(input=text[:16000], model=self.model_name)
        self.total_tokens += response.usage.total_tokens
        return response.data[0].embedding
    
    def compute_embeddings(self, chunks) -> None:
        """Compute embeddings for all code chunks."""
        print("Computing embeddings for all code chunks...")
        for i, chunk in enumerate(chunks):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(chunks)} chunks. Total Cumulative Cost: {self.total_tokens / 1000000 * 0.02}")
            chunk.embedding = self.compute_embedding(chunk.text)
            
    def find_similar(self, query, chunks, top_k=5) :
        """Find top_k similar code chunks to the query code."""
        query_embedding = self.compute_embedding(query)
        
        # Compute cosine similarities
        similarities = []
        for chunk in chunks:
            if chunk.embedding is not None:
                sim = cosine_similarity([query_embedding], [chunk.embedding])[0][0]
                similarities.append((chunk, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

class SearchEngine:
    def __init__(self, repo_path, model_name="text-embedding-3-small"):
        self.repo_path = repo_path 
        self.parser = TextParser()
        self.embedding_engine = EmbeddingEngine(model_name)
        self.chunks = []

    def index_repository(self):
        print(f"Indexing {self.repo_path}")

        index_file = os.path.join(self.repo_path, "paper_index.pkl")
        if os.path.exists(index_file):
            print("Loading existing index...")
            with open(index_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print("Index loaded successfully")
        else:
            print("Creating new index")
            paper_path = os.path.join(self.repo_path, "paper.txt")
            self.chunks = self.parser.parse_file(paper_path)
            self.embedding_engine.compute_embeddings(self.chunks)

            with open(index_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            print("Index saved successfully")
        
        print("Indexing complete")

    def search(self, query, top_k=5):
        return self.embedding_engine.find_similar(query, self.chunks, top_k)

    def display_results(self, results):
        """Display search results in a readable format."""
        print("\n=== SEARCH RESULTS ===")
        res_string = ""
        
        for i, (chunk, similarity) in enumerate(results):
            print(f"\nRESULT #{i+1} - Similarity: {similarity:.4f}")
            res_string += f"{chunk.text}\n\n"
        return res_string
    

if __name__ == "__main__":
    top_k = 5
    paper_ids = ["2303.11932", "2110.03485", "2309.05569", "2205.00048"]

    for paper_id in paper_ids:
        print(f"Paper id: {paper_id}")
        project_dir = os.path.join(this_dir, "MLRC", paper_id)
        code_dir = os.path.join(project_dir, "code")

        engine = SearchEngine(project_dir)
        engine.index_repository()
    
        # Read functions.jsonl file
        functions_file = os.path.join(project_dir, "functions.jsonl")
        functions = []
        if os.path.exists(functions_file):
            with open(functions_file, 'r') as f:
                for line in f:
                    functions.append(json.loads(line))
            print(f"Loaded {len(functions)} functions from {paper_id}")
        
        for function in functions:
            target_function = function["name"]
            target_file = function["file"]
            target_content = []
            with open(os.path.join(code_dir, target_file), 'r') as f:
                for line in f:
                    target_content.append(line)
            
            target_content = "\n".join(target_content[int(function["header_line"])-1:int(function["line_end"])])
            print("Target Content")

            results = engine.search(target_content, top_k)
            paper_context = engine.display_results(results)
            
            function["paper_context_embedding"] = paper_context

        # Write functions back to functions.jsonl
        with open(functions_file, 'w') as f:
            for function in functions:
                json.dump(function, f)
                f.write('\n')
            print(f"Wrote back {len(functions)} functions to {paper_id}")
    
    
