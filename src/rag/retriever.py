import logging
from typing import List, Dict, Any, Optional
from .vector_store import VectorStore
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """文档检索器"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
    
    async def retrieve_documents(self, query: str, n_results: int = 5, 
                               filter_category: Optional[str] = None,
                               min_similarity: float = 0.15) -> List[str]:
        """检索相关文档"""
        try:
            # 记录原始查询
            logger.info(f"开始文档检索，原始查询: '{query}'")
            
            # 构建过滤条件
            filter_dict = None
            if filter_category:
                filter_dict = {"category": filter_category}
                logger.info(f"应用类别过滤: {filter_category}")
            
            # 关键词增强查询
            enhanced_query = query
            if "年假" in query or "休假" in query or "假期" in query:
                enhanced_query += " 公司政策 员工福利 休假制度"
                logger.info(f"查询增强: '{enhanced_query}'")
            
            # 执行向量搜索
            search_results = self.vector_store.search_similar(
                query=enhanced_query,
                n_results=n_results * 4,  # 获取更多结果以便过滤
                filter_dict=filter_dict
            )
            
            # 过滤低相似度结果
            filtered_results = [
                result for result in search_results 
                if result["similarity_score"] >= min_similarity
            ]
            
            # 记录搜索结果详情
            logger.info(f"搜索查询: '{enhanced_query}', 找到 {len(search_results)} 个结果, 过滤后 {len(filtered_results)} 个结果 (阈值: {min_similarity})")
            if search_results:
                for i, result in enumerate(search_results[:5]):
                    logger.info(f"结果 {i+1}: 文件={result['metadata'].get('filename', '未知')}, 类别={result['metadata'].get('category', '未知')}, 相似度={result['similarity_score']:.3f}")
            
            # 提取文档内容
            documents = []
            for result in filtered_results[:n_results]:
                content = result["content"]
                metadata = result["metadata"]
                
                # 添加元数据信息到文档内容
                enriched_content = f"""
文档来源: {metadata.get('filename', '未知')}
文档类别: {metadata.get('category', '未知')}
相似度: {result['similarity_score']:.3f}

内容:
{content}
"""
                documents.append(enriched_content.strip())
            
            logger.info(f"检索到 {len(documents)} 个相关文档，查询: {query}")
            return documents
            
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    async def retrieve_with_rerank(self, query: str, n_results: int = 5,
                                  filter_category: Optional[str] = None) -> List[str]:
        """带重排序的文档检索"""
        try:
            # 首先获取更多候选文档
            candidates = await self.retrieve_documents(
                query=query,
                n_results=n_results * 3,
                filter_category=filter_category,
                min_similarity=0.2  # 降低阈值获取更多候选
            )
            
            if not candidates:
                return []
            
            # TODO: 实现更复杂的重排序逻辑
            # 这里简化处理，按原有相似度排序
            return candidates[:n_results]
            
        except Exception as e:
            logger.error(f"重排序检索失败: {e}")
            return []
    
    def add_documents_from_directory(self, directory_path: str) -> bool:
        """从目录添加文档到向量存储"""
        try:
            # 处理目录中的所有文档
            document_chunks = self.document_processor.process_directory(directory_path)
            
            if not document_chunks:
                logger.warning(f"目录中没有找到可处理的文档: {directory_path}")
                return True
            
            # 添加到向量存储
            success = self.vector_store.add_documents(document_chunks)
            
            if success:
                logger.info(f"成功从目录添加文档: {directory_path}")
            else:
                logger.error(f"从目录添加文档失败: {directory_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"从目录添加文档失败: {e}")
            return False
    
    def get_retriever_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return self.vector_store.get_collection_stats()
