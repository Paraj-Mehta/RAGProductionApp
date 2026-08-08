import torch
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer


EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading embedding model on: {device}")

embedding_model = SentenceTransformer(
    EMBED_MODEL,
    device=device,
)

splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Loading the text data from pdf using PDFReader and chunking them
def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)

    texts = [
        d.text
        for d in docs
        if getattr(d, "text", None)
    ]

    chunks = []

    for text in texts:
        chunks.extend(splitter.split_text(text))

    return chunks

# Embedding the chunks of data and converting into vectors
def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings.tolist()