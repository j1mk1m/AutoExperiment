import os
import ast
import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import glob
import argparse
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Tuple, Any, Optional

this_dir = os.path.dirname(__file__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Chunk:
    def __init__(self, content):
        self.content = content
        self.embedding = None

class CodeChunk(Chunk):
    """Class to represent a code chunk with its metadata.""" 
    def __init__(self, code: str, filepath: str, start_line: int, end_line: int):
        super().__init__(code)
        self.filepath = filepath
        self.start_line = start_line
        self.end_line = end_line
    
    def __repr__(self):
        return f"CodeChunk(file={self.filepath}, lines={self.start_line}-{self.end_line})"

class TextChunk(Chunk):
    def __init__(self, text):
        super().__init__(text)
        self.embedding = None
    
    def __repr__(self):
        return f"TextChunk({self.content[:50]}...{self.content[-50:]})"


class CodeParser:
    """Parser to extract code chunks from Python files."""
    
    def parse_file(self, filepath: str, root_path) -> List[CodeChunk]:
        """Parse a single Python file and extract code chunks."""
        chunks = []
        full_path = os.path.join(root_path, filepath)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as file:
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
                
        except Exception as e:
            print(f"Error parsing {filepath}: {str(e)}")
        
        return chunks
    
    def parse_repository(self, repo_path: str) -> List[CodeChunk]:
        """Parse all Python files in a repository."""
        all_chunks = []
        
        for filepath in glob.glob(os.path.join(repo_path, "**", "*.py"), recursive=True):
            all_chunks.extend(self.parse_file(filepath, repo_path))
            
        print(f"Extracted {len(all_chunks)} code chunks from repository")
        return all_chunks

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
            if i == len(chunks) - 1 or i % 100 == 0 and i > 0:
                print(f"Processed {i}/{len(chunks)} chunks. Total Cumulative Cost: {self.total_tokens / 1000000 * 0.02}")
            chunk.embedding = self.compute_embedding(chunk.content)
            
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


class CodeSearchEngine:
    """Main engine for code similarity search."""
    
    def __init__(self, repo_path: str, llm_manager, model_name: str = "text-embedding-3-small"):
        self.repo_path = repo_path
        self.llm_manager = llm_manager
        self.parser = CodeParser()
        self.embedding_engine = EmbeddingEngine(model_name)
        self.chunks = {}
    
    def get_top_files(self, query):
        _, output_str = traverse_repository(self.repo_path)
        prompt = f"Given repository structure, return top 10 file paths related to the code. Repository structure: \n{output_str}\nCode: \n{query}\n\nReturn only a new line-separated list of paths like this:\n```\n./relative/path/to/file1\n./relative/path/to/file2\n...\n./relative/path/to/file10\n```"
        print(f"### PROMPT ### \n{prompt}")
        response = self.llm_manager.call_llm([{"role": "user", "content": prompt}], None, model="gpt-4o-mini").response.content
        print(f"### Response ###\n{response}")
        filepaths = response.split("```")[1].strip().split("\n")
        real_files = []
        for filepath in filepaths:
            if filepath in self.chunks:
                real_files.append(filepath)
            else:
                chunks = self.parser.parse_file(filepath, self.repo_path)
                if len(chunks) > 0:
                    # if real file, save chunks and compute embeddings
                    self.chunks[filepath] = chunks
                    self.embedding_engine.compute_embeddings(chunks)
                    real_files.append(filepath)
        
        return real_files
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Search for similar code chunks."""
        filepaths = self.get_top_files(query)
        chunks = []
        for filepath in filepaths:
            chunks.extend(self.chunks[filepath])

        results = self.embedding_engine.find_similar(query, chunks, top_k)
        return results, self.display_results(results)
    
    def display_results(self, results: List[Tuple[CodeChunk, float]]) -> None:
        """Display search results in a readable format."""
        print("\n=== SEARCH RESULTS ===")
        res_string = ""
        
        for i, (chunk, similarity) in enumerate(results):
            if i == 0: continue
            print(f"\nRESULT #{i+1} - Similarity: {similarity:.4f}")
            print(f"File: {chunk.filepath} (Lines {chunk.start_line}-{chunk.end_line})")
            res_string += f"File: {chunk.filepath}" + "\n" + "-" * 50 + "\n" + chunk.content + "\n" + "-"*50 + "\n\n"
        return res_string

             
class SearchEngine:
    def __init__(self, repo_path, model_name="text-embedding-3-small"):
        self.repo_path = repo_path 
        self.parser = TextParser()
        self.embedding_engine = EmbeddingEngine(model_name)
        index_file = os.path.join(self.repo_path, "paper_index.pkl")
        if os.path.exists(index_file):
            print("Loading index file")
            with open(index_file, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            print("Creating new index")
            paper_path = os.path.join(self.repo_path, "paper.txt")
            self.chunks = self.parser.parse_file(paper_path)
            self.embedding_engine.compute_embeddings(self.chunks)
 
    def search(self, query, top_k=5):
        results = self.embedding_engine.find_similar(query, self.chunks, top_k)
        return results, self.display_results(results)

    def display_results(self, results):
        """Display search results in a readable format."""
        print("\n=== SEARCH RESULTS ===")
        res_string = ""
        
        for i, (chunk, similarity) in enumerate(results):
            print(f"\nRESULT #{i+1} - Similarity: {similarity:.4f}")
            res_string += f"{chunk.content}\n\n"
        return res_string
     
    
def traverse_repository(repo_path):
    """
    Traverse a repository and return the directory structure for .py files only.
    
    Args:
        repo_path (str): Path to the repository to traverse
        output_file (str, optional): Path to save the output. If None, prints to console.
    
    Returns:
        dict: A nested dictionary representing the directory structure
    """
    # Convert to absolute path and resolve any symlinks
    repo_path = os.path.abspath(os.path.expanduser(repo_path))
    
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path '{repo_path}' does not exist")
    
    # Store the directory structure
    structure = {}
    
    # Get the base directory name for later reference
    base_dir_name = os.path.basename(repo_path)
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories (those starting with .)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Filter for .py files only
        py_files = [f for f in files if f.endswith('.py')]
        
        if py_files:
            # Get relative path from the repo root
            rel_path = os.path.relpath(root, repo_path)
            if rel_path == '.':
                rel_path = ''
            
            # Build the path components
            path_parts = rel_path.split(os.sep) if rel_path else []
            
            # Navigate to the correct position in the structure
            current = structure
            for part in path_parts:
                if part:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
            
            # Add the Python files at this level
            for py_file in py_files:
                current[py_file] = None  # None indicates this is a file, not a directory
    
    # Output the structure
    output_str = format_structure(structure, base_dir_name)
 
    return structure, output_str


def format_structure(structure, root_name, indent_level=0):
    """
    Format the directory structure as a string.
    
    Args:
        structure (dict): The directory structure
        root_name (str): The name of the root directory
        indent_level (int): Current indentation level
    
    Returns:
        str: Formatted string representation of the directory structure
    """
    result = []
    indent = "  " * indent_level
    
    if indent_level == 0:
        result.append(f"{root_name}/")
    
    for name, contents in sorted(structure.items(), key=lambda x: (x[1] is not None, x[0])):
        if contents is None:  # This is a file
            result.append(f"{indent}  {name}")
        else:  # This is a directory
            result.append(f"{indent}  {name}/")
            result.append(format_structure(contents, "", indent_level + 1))
    
    return "\n".join(result)