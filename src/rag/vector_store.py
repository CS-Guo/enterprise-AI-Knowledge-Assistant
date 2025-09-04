import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json

from langchain_openai import OpenAIEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """向量存储管理器"""
    
    def __init__(self, collection_name: str = "enterprise_docs"):
        self.collection_name = collection_name
        self.embedding_model = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.base_url,
            model=settings.embedding_model
        )
        
        # 初始化ChromaDB
        os.makedirs(settings.vector_db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"加载现有集合: {collection_name}")
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "企业文档向量集合"}
            )
            logger.info(f"创建新集合: {collection_name}")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入向量"""
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"嵌入向量生成失败: {e}")
            return []

    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """添加文档到向量存储"""
        try:
            if not documents:
                logger.warning("没有文档需要添加")
                return True
            
            logger.info(f"开始添加 {len(documents)} 个文档块到向量存储")
            
            # 提取文本和元数据
            texts = [doc["chunk_text"] for doc in documents]
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                # 创建唯一ID
                doc_id = f"{doc['filename']}_{doc['chunk_id']}_{i}"
                ids.append(doc_id)
                
                # 准备元数据（ChromaDB要求字符串、数字或布尔值）
                metadata = {
                    "filename": doc["filename"],
                    "file_path": doc["file_path"],
                    "chunk_id": str(doc["chunk_id"]),
                    "category": doc["category"],
                    "file_extension": doc["file_extension"],
                    "chunk_length": doc["chunk_length"],
                    "created_time": str(doc["created_time"]),
                    "modified_time": str(doc["modified_time"])
                }
                metadatas.append(metadata)
            
            # 记录一些文档信息
            if documents:
                sample_doc = documents[0]
                logger.info(f"文档示例 - 文件名: {sample_doc.get('filename', '未知')}, 类别: {sample_doc.get('category', '未知')}")
                logger.info(f"文档块长度范围: {min(len(doc.get('chunk_text', '')) for doc in documents)} - {max(len(doc.get('chunk_text', '')) for doc in documents)} 字符")
            
            # 生成嵌入向量
            embeddings = self.generate_embeddings(texts)
            if not embeddings:
                logger.error("嵌入向量生成失败")
                return False
            
            # 添加到集合
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )
            
            # 获取集合统计信息
            stats = self.get_collection_stats()
            logger.info(f"成功添加 {len(documents)} 个文档块到向量存储，当前总文档数: {stats.get('total_documents', '未知')}")
            return True
            
        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {e}")
            return False

    def search_similar(self, query: str, n_results: int = 5, 
                      filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """搜索相似文档"""
        try:
            # 生成查询嵌入
            query_embedding_list = self.generate_embeddings([query])
            if not query_embedding_list:
                logger.error("查询嵌入生成失败，返回空结果")
                return []
            query_embedding = query_embedding_list[0]
            
            # 记录查询信息
            logger.info(f"执行向量搜索，查询: '{query}', 请求结果数: {n_results}")
            if filter_dict:
                logger.info(f"应用过滤条件: {filter_dict}")
            
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # 格式化结果
            formatted_results = []
            if results["documents"] and len(results["documents"]) > 0:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                
                logger.info(f"ChromaDB返回了 {len(documents)} 个结果")
                
                for doc, metadata, distance in zip(documents, metadatas, distances):
                    similarity_score = 1 - distance  # 转换为相似度分数
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": similarity_score,
                        "distance": distance
                    })
                    
            else:
                logger.warning(f"ChromaDB没有找到匹配的文档，查询: '{query}'")
                # 如果没有找到结果且有过滤条件，尝试不带过滤条件再次查询
                if filter_dict:
                    logger.info("尝试不带过滤条件再次查询")
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=n_results,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    if results["documents"] and len(results["documents"]) > 0:
                        documents = results["documents"][0]
                        metadatas = results["metadatas"][0]
                        distances = results["distances"][0]
                        
                        logger.info(f"不带过滤条件查询返回 {len(documents)} 个结果")
                        
                        for doc, metadata, distance in zip(documents, metadatas, distances):
                            similarity_score = 1 - distance
                            formatted_results.append({
                                "content": doc,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance
                            })
            
            # 记录一些结果信息
            if formatted_results:
                logger.info(f"前3个结果的相似度: {[r['similarity_score'] for r in formatted_results[:3]]}") 
                logger.info(f"前3个结果的文件名: {[r['metadata'].get('filename', '未知') for r in formatted_results[:3]]}")
            
            logger.info(f"搜索完成，返回 {len(formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": settings.embedding_model
            }
        except Exception as e:
            logger.error(f"获取集合统计失败: {e}")
            return {}
    
    def delete_documents(self, filter_dict: Dict[str, Any]) -> bool:
        """根据过滤条件删除文档"""
        try:
            # 先查询要删除的文档
            results = self.collection.query(
                query_embeddings=[[0] * 384],  # 占位符嵌入
                n_results=1000,  # 获取所有匹配的文档
                where=filter_dict,
                include=["metadatas"]
            )
            
            if results["ids"] and len(results["ids"]) > 0:
                ids_to_delete = results["ids"][0]
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"删除了 {len(ids_to_delete)} 个文档")
                return True
            else:
                logger.info("没有找到匹配的文档需要删除")
                return True
                
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """清空集合"""
        try:
            # 删除集合
            self.client.delete_collection(name=self.collection_name)
            # 重新创建集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "企业文档向量集合"}
            )
            logger.info(f"集合 {self.collection_name} 已清空")
            return True
        except Exception as e:
            logger.error(f"清空集合失败: {e}")
            return False