import os
import ast
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


class CodeChunk:
    """Class to represent a code chunk with its metadata."""
    
    def __init__(self, code: str, filepath: str, start_line: int, end_line: int):
        self.code = code
        self.filepath = filepath
        self.start_line = start_line
        self.end_line = end_line
        self.embedding = None
    
    def __repr__(self):
        return f"CodeChunk(file={self.filepath}, lines={self.start_line}-{self.end_line})"


class CodeParser:
    """Parser to extract code chunks from Python files."""
    
    def parse_file(self, filepath: str) -> List[CodeChunk]:
        """Parse a single Python file and extract code chunks."""
        chunks = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                
            tree = ast.parse(content)
            
            # Extract functions and methods
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    chunk_code = ast.get_source_segment(content, node) 
                    if chunk_code:
                        chunks.append(CodeChunk(
                            code=chunk_code,
                            filepath=filepath,
                            start_line=node.lineno,
                            end_line=node.end_lineno
                        ))
            
            # If no functions/classes were found, add the whole file as a chunk
            if not chunks and content.strip():
                chunks.append(CodeChunk(
                    code=content,
                    filepath=filepath,
                    start_line=1,
                    end_line=len(content.splitlines())
                ))
                
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Error parsing {filepath}: {str(e)}")
        
        return chunks
    
    def parse_repository(self, repo_path: str) -> List[CodeChunk]:
        """Parse all Python files in a repository."""
        all_chunks = []
        
        for filepath in glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True):
            all_chunks.extend(self.parse_file(filepath))
            
        print(f"Extracted {len(all_chunks)} code chunks from repository")
        return all_chunks


class EmbeddingEngine:
    """Engine to compute and compare embeddings for code chunks."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """Initialize with a sentence transformer model."""
        self.model_name = model_name
        self.total_tokens = 0
        
    def compute_embedding(self, code: str) -> np.ndarray:
        """Compute embedding for a single code snippet."""
        try:
            response = client.embeddings.create(input=code, model=self.model_name)
        except Exception as e:
            print(e)
            response = client.embeddings.create(input=code[:16000], model=self.model_name)
        self.total_tokens += response.usage.total_tokens
        return response.data[0].embedding
    
    def compute_embeddings(self, chunks: List[CodeChunk]) -> None:
        """Compute embeddings for all code chunks."""
        print("Computing embeddings for all code chunks...")
        for i, chunk in enumerate(chunks):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(chunks)} chunks. Total Cumulative Cost: {self.total_tokens / 1000000 * 0.02}")
            chunk.embedding = self.compute_embedding(chunk.code)
            
    def find_similar(self, query_code: str, chunks: List[CodeChunk], top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Find top_k similar code chunks to the query code."""
        query_embedding = self.compute_embedding(query_code)
        
        # Compute cosine similarities
        similarities = []
        for chunk in chunks:
            if chunk.embedding is not None:
                sim = cosine_similarity([query_embedding], [chunk.embedding])[0][0]
                similarities.append((chunk, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class CodeSearchEngine:
    """Main engine for code similarity search."""
    
    def __init__(self, repo_path: str, model_name: str = "text-embedding-3-small"):
        self.repo_path = repo_path
        self.parser = CodeParser()
        self.embedding_engine = EmbeddingEngine(model_name)
        self.chunks = []
        
    def index_repository(self) -> None:
        """Index the repository by parsing and computing embeddings."""
        print(f"Indexing repository: {self.repo_path}")
        
        # Check if index file exists
        index_file = os.path.join(self.repo_path, "code_index.pkl")
        if os.path.exists(index_file):
            print("Loading existing index...")
            with open(index_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print("Index loaded successfully!")
        else:
            print("Creating new index...")
            self.chunks = self.parser.parse_repository(os.path.join(self.repo_path, "code"))
            self.embedding_engine.compute_embeddings(self.chunks)
            
            # Save index to file
            with open(index_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            print("Index saved successfully!")
            
        print("Repository indexing complete!")
        
    def search(self, query_code: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Search for similar code chunks."""
        return self.embedding_engine.find_similar(query_code, self.chunks, top_k)
    
    def display_results(self, results: List[Tuple[CodeChunk, float]]) -> None:
        """Display search results in a readable format."""
        print("\n=== SEARCH RESULTS ===")
        res_string = ""
        
        for i, (chunk, similarity) in enumerate(results):
            if i == 0: continue
            print(f"\nRESULT #{i+1} - Similarity: {similarity:.4f}")
            print(f"File: {chunk.filepath} (Lines {chunk.start_line}-{chunk.end_line})")
            # print("-" * 50)
            # print(chunk.code)
            # print("-" * 50)
            res_string += f"File: {chunk.filepath.split('code/')[1]}" + "\n" + "-" * 50 + "\n" + chunk.code + "\n" + "-"*50 + "\n\n"
        return res_string
                

if __name__ == "__main__":
    top_k = 5
    paper_ids = ["2303.11932", "2110.03485", "2309.05569", "2205.00048"]

    for paper_id in paper_ids:
        print(f"Paper id: {paper_id}")
        project_dir = os.path.join(this_dir, "MLRC", paper_id)
        code_dir = os.path.join(project_dir, "code")

        engine = CodeSearchEngine(project_dir)
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
            # print("-" * 50)
            # print(target_content)
            # print("-" * 50)

            results = engine.search(target_content, top_k)
            code_context = engine.display_results(results)
            
            function["code_context_embedding"] = code_context

        # Write functions back to functions.jsonl
        with open(functions_file, 'w') as f:
            for function in functions:
                json.dump(function, f)
                f.write('\n')
            print(f"Wrote back {len(functions)} functions to {paper_id}")
    
    
