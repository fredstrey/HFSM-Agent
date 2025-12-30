"""
PDF Processor usando Docling para extrair e processar PDFs
"""
import os
from typing import List, Dict, Optional
from pathlib import Path

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat

from embedding_manager.embedding_manager import EmbeddingManager


class PDFProcessor:
    """Processa PDFs usando Docling e armazena no Qdrant"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Inicializa o processador de PDF
        
        Args:
            embedding_manager: Gerenciador de embeddings e Qdrant
        """
        self.embedding_manager = embedding_manager
        self._converter = None  # Lazy loading
    
    @property
    def converter(self):
        """Lazy loading do DocumentConverter para evitar consumo de recursos desnecessário"""
        if self._converter is None:
            print("🔧 Inicializando Docling DocumentConverter (primeira vez)...")
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
            print("✅ Docling inicializado!")
        return self._converter
    
    def process_pdf(
        self,
        pdf_path: str,
        max_tokens: int = 500,
    ) -> Dict:
        """
        Processa um PDF e armazena no Qdrant
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            max_tokens: Máximo de tokens por chunk
            overlap_tokens: Tokens de sobreposição (não usado no Docling)
            
        Returns:
            Dicionário com estatísticas do processamento
        """
        print(f"📄 Processando PDF: {pdf_path}")
        
        # Verifica se arquivo existe
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
        
        pdf_file_name = Path(pdf_path).stem
        
        # Converte PDF com Docling
        print("🔄 Convertendo PDF com Docling...")
        result = self.converter.convert(source=pdf_path)
        doc = result.document
        
        # Cria chunker híbrido
        print(f"✂️  Criando chunks (max {max_tokens} tokens)...")
        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=max_tokens
        )
        
        # Gera chunks
        chunks_list = []
        metadatas = []
        
        for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
            # Usa chunk.text diretamente (não contextualize que causa erro com DocItem)
            chunk_text = chunk.text
            chunks_list.append(chunk_text)
            
            # Cria metadados básicos
            metadata = {
                "source": pdf_file_name,
                "chunk_id": f"{pdf_file_name}_chunk_{i:04d}",
                "total_chunks": -1,  # Será atualizado depois
                "chunk_index": i,
                "doc_type": "pdf"
            }
            metadatas.append(metadata)
        
        total_chunks = len(chunks_list)
        
        # Atualiza total_chunks em todos os metadados
        for metadata in metadatas:
            metadata["total_chunks"] = total_chunks
        
        print(f"✅ Criados {total_chunks} chunks")
        
        # Adiciona ao Qdrant
        print("💾 Adicionando chunks ao Qdrant...")
        self.embedding_manager.add_documents(chunks_list, metadatas)
        
        # Calcula estatísticas
        # Nota: Docling não expõe contagem de tokens diretamente,
        # então estimamos baseado no tamanho do texto
        total_chars = sum(len(chunk) for chunk in chunks_list)
        avg_chars = total_chars / total_chunks if total_chunks > 0 else 0
        
        # Estimativa: ~4 chars por token (aproximação)
        estimated_total_tokens = total_chars // 4
        estimated_avg_tokens = avg_chars / 4
        
        stats = {
            "pdf_file": pdf_file_name,
            "total_chunks": total_chunks,
            "total_tokens": estimated_total_tokens,
            "avg_tokens_per_chunk": estimated_avg_tokens,
            "max_tokens_per_chunk": max_tokens,
            "min_tokens_per_chunk": min(len(chunk) // 4 for chunk in chunks_list) if chunks_list else 0
        }
        
        print(f"📊 Processamento concluído: {stats}")
        
        return stats
