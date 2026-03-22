"""QueryAgent - Web search and information gathering agent.

Responsible for:
1. Analyzing user query to generate search strategies
2. Performing multi-round searches (web + news)
3. Summarizing and structuring search findings
4. Posting key findings to the forum for other agents
"""

import logging
from agents.base_agent import BaseAgent
from tools.search import search_web, search_news, extract_key_info
from tools.text_analysis import extract_keywords
from config import Config

logger = logging.getLogger(__name__)


class QueryAgent(BaseAgent):
    """Agent specialized in web search and information gathering."""

    def __init__(self):
        super().__init__(
            name="QueryAgent",
            role="信息搜索专家",
            description="负责从互联网搜索引擎获取与用户查询相关的最新信息和新闻报道",
        )

    def _generate_search_queries(self, user_query: str) -> list[str]:
        """Use LLM to generate diverse search queries from user input."""
        self.update_status("thinking", "正在分析查询，制定搜索策略...", 0.1)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个搜索策略专家。根据用户的舆情分析需求，生成5个不同角度的搜索查询词。"
                    "包括：事件概述、各方观点、时间线、社会反应、专家评价等角度。"
                    "请直接返回JSON数组格式，例如：[\"查询1\", \"查询2\", ...]"
                ),
            },
            {
                "role": "user",
                "content": f"用户的舆情分析需求: {user_query}\n\n请生成5个搜索查询词（JSON数组）：",
            },
        ]

        try:
            result = self.chat_json(messages)
            if isinstance(result, list):
                return result[:5]
            return [user_query]
        except Exception:
            # Fallback: use the original query + some variants
            return [
                user_query,
                f"{user_query} 最新消息",
                f"{user_query} 社会评价",
                f"{user_query} 专家观点",
                f"{user_query} 发展趋势",
            ]

    def _search_and_collect(self, queries: list[str]) -> dict:
        """Execute searches for all queries and collect results."""
        self.update_status("searching", "正在搜索互联网信息...", 0.3)

        all_web = []
        all_news = []
        seen_urls = set()

        for i, query in enumerate(queries):
            self.update_status(
                "searching",
                f"正在搜索 ({i+1}/{len(queries)}): {query}",
                0.3 + (i / len(queries)) * 0.3,
            )

            # Web search
            web_results = search_web(query, max_results=Config.SEARCH_MAX_RESULTS)
            for r in web_results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_web.append(r)

            # News search
            news_results = search_news(query, max_results=Config.SEARCH_MAX_RESULTS)
            for r in news_results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_news.append(r)

        return {
            "web_results": all_web,
            "news_results": all_news,
            "total_sources": len(all_web) + len(all_news),
        }

    def _analyze_results(self, user_query: str, search_data: dict) -> dict:
        """Use LLM to analyze and summarize search results."""
        self.update_status("analyzing", "正在分析搜索结果...", 0.7)

        web_text = extract_key_info(search_data["web_results"][:15])
        news_text = extract_key_info(search_data["news_results"][:15])

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的舆情分析师。请根据搜索结果，对用户关注的话题进行全面分析。"
                    "请使用JSON格式返回分析结果，包含以下字段：\n"
                    "{\n"
                    '  "summary": "事件/话题概述（200-300字）",\n'
                    '  "key_events": ["关键事件1", "关键事件2", ...],\n'
                    '  "stakeholders": ["相关方1", "相关方2", ...],\n'
                    '  "public_opinion": "舆论倾向分析（100-200字）",\n'
                    '  "timeline": [{"time": "时间", "event": "事件"}],\n'
                    '  "hot_topics": ["热点话题1", "热点话题2", ...],\n'
                    '  "risk_assessment": "风险评估（100字）",\n'
                    '  "trend_prediction": "趋势预测（100字）"\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"分析主题: {user_query}\n\n"
                    f"--- 网页搜索结果 ---\n{web_text}\n\n"
                    f"--- 新闻搜索结果 ---\n{news_text}"
                ),
            },
        ]

        try:
            analysis = self.chat_json(messages)
        except Exception:
            analysis = {
                "summary": "分析过程中出现错误，请检查LLM配置。",
                "key_events": [],
                "stakeholders": [],
                "public_opinion": "无法分析",
                "timeline": [],
                "hot_topics": [],
                "risk_assessment": "无法评估",
                "trend_prediction": "无法预测",
            }

        return analysis

    def run(self, query: str, forum_context: list[dict] | None = None) -> dict:
        """Execute the full search and analysis workflow."""
        self.update_status("starting", f"开始搜索分析: {query}", 0.0)

        try:
            # Step 1: Generate diverse search queries
            search_queries = self._generate_search_queries(query)
            self.emit_event("search_queries", {"queries": search_queries})

            # Step 2: Execute searches
            search_data = self._search_and_collect(search_queries)

            # Step 3: Record sources as findings
            for item in search_data["web_results"][:10]:
                self.add_finding({
                    "type": "web_source",
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                })

            for item in search_data["news_results"][:10]:
                self.add_finding({
                    "type": "news_source",
                    "title": item["title"],
                    "url": item["url"],
                    "snippet": item["snippet"],
                })

            # Step 4: Analyze results
            analysis = self._analyze_results(query, search_data)

            # Step 5: Post key findings to forum
            forum_summary = (
                f"## QueryAgent 搜索分析报告\n\n"
                f"共搜索到 {search_data['total_sources']} 条信息来源\n\n"
                f"**概述**: {analysis.get('summary', 'N/A')}\n\n"
                f"**舆论倾向**: {analysis.get('public_opinion', 'N/A')}\n\n"
                f"**热点话题**: {', '.join(analysis.get('hot_topics', []))}\n\n"
                f"**趋势预测**: {analysis.get('trend_prediction', 'N/A')}"
            )
            self.post_to_forum(forum_summary, topic="搜索分析结果")

            self.update_status("done", "搜索分析完成", 1.0)

            return {
                "analysis": analysis,
                "sources": {
                    "web": search_data["web_results"][:10],
                    "news": search_data["news_results"][:10],
                },
                "search_queries": search_queries,
                "total_sources": search_data["total_sources"],
            }

        except Exception as e:
            self.update_status("error", f"搜索分析失败: {str(e)}")
            return {"error": str(e)}
