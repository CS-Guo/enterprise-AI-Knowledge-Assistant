import logging
from typing import List, Dict, Any, Optional, Tuple
import math
import re
from .vector_store import VectorStore
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """轻量分词：
    - 英文/数字：按非单词字符切分为词
    - CJK：生成字符bigram以增强匹配鲁棒性
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip().lower()
    # 提取ASCII词
    ascii_tokens = re.findall(r"[a-z0-9_]+", text)
    # 提取CJK字符序列并生成bigram
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = ["".join(cjk_chars[i:i+2]) for i in range(len(cjk_chars)-1)] if len(cjk_chars) > 1 else cjk_chars
    return ascii_tokens + cjk_bigrams


def _lexical_scores(query: str, docs: List[str]) -> List[float]:
    """计算简化BM25风格的词法匹配分数（基于TF-IDF近似）。"""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return [0.0] * len(docs)
    # 文档tokens
    doc_tokens_list: List[List[str]] = [_tokenize(d) for d in docs]
    N = len(docs)
    # 计算DF
    df: Dict[str, int] = {}
    for tokens in doc_tokens_list:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    # 计算IDF
    idf: Dict[str, float] = {t: math.log((N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}
    # 计算query tf
    q_tf: Dict[str, int] = {}
    for t in q_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1
    # 计算每篇文档分数
    scores: List[float] = []
    for tokens in doc_tokens_list:
        d_tf: Dict[str, int] = {}
        for t in tokens:
            d_tf[t] = d_tf.get(t, 0) + 1
        s = 0.0
        for t, qf in q_tf.items():
            if t in d_tf and t in idf:
                s += (qf * d_tf[t]) * idf[t]
        scores.append(s)
    # 归一化到[0,1]
    if not scores:
        return scores
    max_s = max(scores)
    if max_s <= 0:
        return [0.0] * len(scores)
    return [s / max_s for s in scores]


class DocumentRetriever:
    """文档检索器"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
    
    async def retrieve_documents(self, query: str, n_results: int = 5, 
                               filter_category: Optional[str] = None,
                               min_similarity: float = 0.15) -> List[str]:
        """检索相关文档（向量检索）"""
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
        """带重排序/混合检索：在原有基础上加入词法信号并融合排序"""
        return await self.retrieve_hybrid(
            query=query,
            n_results=n_results,
            intent_category=filter_category,
        )

    async def retrieve_hybrid(self, query: str, n_results: int = 5,
                              intent_category: Optional[str] = None) -> List[str]:
        """混合检索（Embedding + 词法融合 + 跨类别合并 + 动态topK/阈值）"""
        try:
            # 动态参数（结合意图与查询长度）
            q_tokens = _tokenize(query)
            qlen = len(q_tokens)
            base_topk = 10 if intent_category in {"hr", "policy"} else (8 if intent_category in {"tech"} else 6)
            expand_topk = 24  # 候选池规模（向量）
            min_sim = 0.13 if intent_category in {"general", None} else 0.15
            # 对短查询提高词法阈值，弱化向量阈值
            lex_min = 0.35 if qlen <= 2 else (0.28 if qlen <= 5 else 0.22)
            alpha = 0.55 if intent_category in {"tech"} else 0.5  # 向量占比

            # 多路向量检索：全局 + 意图类别 + 相关类别
            pools: List[Tuple[str, List[Dict[str, Any]]]] = []
            # 1) 无筛选全局
            global_results = self.vector_store.search_similar(query=query, n_results=expand_topk, filter_dict=None)
            pools.append(("global", global_results))
            # 2) 意图类别
            if intent_category:
                cat_results = self.vector_store.search_similar(query=query, n_results=expand_topk, filter_dict={"category": intent_category})
                pools.append((intent_category, cat_results))
            # 3) 相关类别补充
            for cat in ["policy", "faq"]:
                if intent_category and cat == intent_category:
                    continue
                try:
                    extra = self.vector_store.search_similar(query=query, n_results=expand_topk, filter_dict={"category": cat})
                    if extra:
                        pools.append((cat, extra))
                except Exception:
                    pass

            # 词法检索候选池：基于 list_documents（小规模场景）
            lexical_candidates: List[Dict[str, Any]] = []
            def _gather_lex_pool(filter_dict: Optional[Dict[str, Any]], cap: int, tag: str):
                try:
                    items = self.vector_store.list_documents(filter_dict=filter_dict, limit=cap)
                    if not items:
                        return
                    texts = [it.get("content", "") for it in items]
                    scores = _lexical_scores(query, texts)
                    # 选取前若干（避免过多噪声）
                    ranked = sorted(zip(items, scores), key=lambda x: x[1], reverse=True)[: max(20, n_results * 4)]
                    for (it, ls) in ranked:
                        # 仅保留必要字段，并标注lex-only
                        lexical_candidates.append({
                            "content": it.get("content", ""),
                            "metadata": it.get("metadata", {}) or {},
                            "similarity_score": 0.0,  # 无向量分
                            "_lex_score": ls,
                            "_source": f"lexical:{tag}",
                        })
                except Exception:
                    pass
            # 全局 + 意图类别 + 相关类别
            _gather_lex_pool(None, 300, "global")
            if intent_category:
                _gather_lex_pool({"category": intent_category}, 200, intent_category)
            for cat in ["policy", "faq"]:
                if intent_category and cat == intent_category:
                    continue
                _gather_lex_pool({"category": cat}, 150, cat)

            # 合并候选，去重（以内容hash+文件名+chunk_id去重）
            seen = set()
            candidates: List[Dict[str, Any]] = []
            # 向量候选
            for tag, results in pools:
                for r in results:
                    content = r.get("content", "")
                    meta = (r.get("metadata") or {})
                    key = (hash(content), meta.get("filename", ""), meta.get("chunk_id", ""))
                    if key in seen:
                        continue
                    seen.add(key)
                    r_copy = dict(r)
                    r_copy["_source"] = f"vector:{tag}"
                    candidates.append(r_copy)
            # 词法候选
            for r in lexical_candidates:
                content = r.get("content", "")
                meta = (r.get("metadata") or {})
                key = (hash(content), meta.get("filename", ""), meta.get("chunk_id", ""))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(r)

            if not candidates:
                return []

            # 统一计算词法得分（对向量候选也计算，用于融合）
            doc_texts = [c.get("content", "") for c in candidates]
            global_lex_scores = _lexical_scores(query, doc_texts)

            # 融合分数并过滤
            fused = []
            kept_by_reason = {"vector": 0, "lexical": 0}
            for idx, (c, lex) in enumerate(zip(candidates, global_lex_scores)):
                emb = float(c.get("similarity_score", 0.0))
                # 对于纯词法候选，使用其预计算词法分的较大值
                lex_final = max(lex, float(c.get("_lex_score", 0.0)))
                score = alpha * emb + (1 - alpha) * lex_final
                # 过滤：向量或词法满足其一即可
                pass_filter = (emb >= min_sim) or (lex_final >= lex_min)
                if not pass_filter:
                    continue
                c_copy = dict(c)
                c_copy["fused_score"] = score
                c_copy["_lex"] = lex_final
                if emb >= min_sim:
                    kept_by_reason["vector"] += 1
                else:
                    kept_by_reason["lexical"] += 1
                fused.append(c_copy)

            if not fused:
                logger.info(f"混合检索过滤后无结果（min_sim={min_sim}, lex_min={lex_min}），返回空")
                return []

            fused.sort(key=lambda x: x.get("fused_score", 0.0), reverse=True)

            # 动态topK：若前k平均分过低，则减少k；若全部来自词法且数量多，则限制为较小k防止噪声
            k = base_topk
            top_slice = fused[:k]
            if top_slice:
                avg_top = sum(x.get("fused_score", 0.0) for x in top_slice) / len(top_slice)
                if avg_top < 0.25 and k > 4:
                    k = 4
                # 如果保留项中词法占比过高而向量弱，进一步收紧k
                lexical_ratio = sum(1 for x in top_slice if x.get("similarity_score", 0.0) < min_sim) / len(top_slice)
                if lexical_ratio > 0.7 and k > 5:
                    k = 5
            selected = fused[: min(n_results, k)]

            # 组装输出
            documents: List[str] = []
            for r in selected:
                content = r.get("content", "")
                metadata = r.get("metadata", {})
                enriched_content = f"""
文档来源: {metadata.get('filename', '未知')}
文档类别: {metadata.get('category', '未知')}
相似度: {r.get('similarity_score', 0.0):.3f}
融合分: {r.get('fused_score', 0.0):.3f}
词法分: {r.get('_lex', 0.0):.3f}
来源: {r.get('_source', 'unknown')}

内容:
{content}
"""
                documents.append(enriched_content.strip())

            logger.info(
                f"混合检索完成，候选={len(candidates)}，保留={len(fused)}(vec={kept_by_reason['vector']},lex={kept_by_reason['lexical']})，返回={len(documents)}，intent={intent_category}, qlen={qlen}"
            )
            return documents
        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
