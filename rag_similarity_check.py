import os
import sys
import json
import ollama

from coloring_logger import logger

class RAGSimilarityCheck():
  
  def __init__(self, config=None, database_folder=None, embedding_model=None):
      
    self.config = config
    self.loaded_entries = []
    self.vector_db = []
    
    if config:
      self.database_folder = self.config["nativa_gpt"]["database_folder"]
      self.embedding_model = self.config["nativa_gpt"]["embedding_model"]
    else:
      self.database_folder = database_folder
      self.embedding_model = embedding_model
    

    if self.read_database_files():
      self.add_chunks_to_database()

  def read_database_files(self):
    self.loaded_entries = []
    
    for root, dirs, files in os.walk(self.database_folder):
      for filename in files:
        file_path = os.path.join(root, filename)
        try:
          with open(file_path, 'r', encoding='utf-8') as file:
            # Check if it's a JSON file
            if filename.endswith('.json'):
              data = json.load(file)
              # Handle both single objects and arrays
              if isinstance(data, list):
                self.loaded_entries.extend(data)
                logger.info(f"Loaded {len(data)} JSON entries from {file_path}")
              else:
                self.loaded_entries.append(data)
                logger.info(f"Loaded 1 JSON entry from {file_path}")
            else:
              # Handle text files as before
              lines = file.readlines()
              self.loaded_entries.extend([line.strip() for line in lines if line.strip()])
              logger.info(f"Loaded {len(lines)} text entries from {file_path}")
            
        except json.JSONDecodeError as e:
          logger.error(f"Failed to parse JSON in {file_path}: {e}")
          return False
        except Exception as e:
          logger.error(f"Failed to read {file_path}: {e}")
          return False
    
    logger.info(f"Total loaded entries: {len(self.loaded_entries)}")
    return True

  def add_chunks_to_database(self):
    for i, entry in enumerate(self.loaded_entries):
      # Create searchable text from JSON object
      if isinstance(entry, dict):
        # Extract searchable text from the JSON object
        searchable_text = self._extract_searchable_text(entry)
      else:
        # It's already a string
        searchable_text = str(entry)
      
      # Create embedding from the searchable text
      embedding = ollama.embed(model=self.embedding_model, input=searchable_text)['embeddings'][0]
      
      # Store the original entry (JSON object or string) with its embedding
      self.vector_db.append((entry, embedding, searchable_text))
      logger.info(f'Added entry {i+1}/{len(self.loaded_entries)} to the database')

  def _extract_searchable_text(self, json_obj, extract_type="word"):
    """Extract searchable text from JSON object"""
    if isinstance(json_obj, dict):
      # For your function format, extract name and description
      if 'function' in json_obj:
        func = json_obj['function']
        return f"{func.get('name', '')} {func.get('description', '')} {func.get('command', '')}"
      else:
        # Generic approach: concatenate all string values
        text_parts = []
        for key, value in json_obj.items():
          if isinstance(value, str) and extract_type == "text":
            text_parts.append(value)
          elif isinstance(value, dict) and extract_type == "dict":
            text_parts.append(self._extract_searchable_text(value))
        return ' '.join(text_parts)
    else:
      return str(json_obj)

  def _get_cosine_similarity(self, a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)

  def retrieve(self, query, top_n=3):
    query_embedding = ollama.embed(model=self.embedding_model, input=query)['embeddings'][0]
    similarities = []
    
    for entry, embedding, searchable_text in self.vector_db:
      similarity = self._get_cosine_similarity(query_embedding, embedding)
      similarities.append((entry, similarity, searchable_text))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

if __name__ == "__main__":
    
    # config_manager = ConfigManager(config_path="/home/pedrodias/Documents/git-repos/NativaGPT/config/config_default.json")
    # config = config_manager.get()

    database_folder = "/home/pedrodias/Documents/git-repos/project-ss-hri/evaluation_metrics"
    embedding_model = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf"

    rag = RAGSimilarityCheck(database_folder=database_folder,
                             embedding_model=embedding_model)

    input_query = input('Ask me a question: ')
    retrieved_knowledge = rag.retrieve(input_query)
    
    logger.info('Retrieved knowledge:')
    for i, (entry, similarity, searchable_text) in enumerate(retrieved_knowledge):
      if i >= 3:  # Stop after 3 results
          break
      logger.info(f' - (similarity: {similarity:.2f})')
      logger.info(f'   Text: {entry}')