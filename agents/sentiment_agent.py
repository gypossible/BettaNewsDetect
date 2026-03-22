"""SentimentAgent - Sentiment analysis and opinion mining agent.

Responsible for:
1. Analyzing sentiment of collected text data
2. Classifying opinions into positive/negative/neutral
3. Identifying key opinion holders and viewpoints
4. Generating sentiment trend analysis
"""

import logging
from agents.base_agent import BaseAgent
from tools.text_analysis import extract_keywords

logger = logging.getLogger(__name__)


class SentimentAgent(BaseAgent):
    """Agent specialized in sentiment analysis and opinion mining."""

    def __init__(self):
        super().__init__(
            name="SentimentAgent",
            role="情感分析专家",
            description="负责对搜集到的信息进行情感分析、观点挖掘和舆论趋势判断",
        )

    def _analyze_sentiment_batch(self, texts: list[dict]) -> dict:
        """Analyze sentiment for a batch of text items using LLM."""
        self.update_status("analyzing", "正在进行情感分析...", 0.3)

        text_block = "\n\n".join([
            f"[{i+1}] 标题: {t.get('title', 'N/A')}\n内容: {t.get('snippet', t.get('content', 'N/A'))}"
            for i, t in enumerate(texts[:20])
        ])

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的舆情情感分析师。请对以下文本逐条分析情感倾向。"
                    "返回JSON格式：\n"
                    "{\n"
                    '  "items": [\n'
                    '    {"id": 1, "sentiment": "positive/negative/neutral", "score": 0.0-1.0, "key_opinion": "核心观点摘要"}\n'
                    "  ],\n"
                    '  "overall": {\n'
                    '    "positive_ratio": 0.0-1.0,\n'
                    '    "negative_ratio": 0.0-1.0,\n'
                    '    "neutral_ratio": 0.0-1.0,\n'
                    '    "dominant_sentiment": "positive/negative/neutral",\n'
                    '    "sentiment_summary": "整体情感倾向分析（150字）"\n'
                    "  }\n"
                    "}"
                ),
            },
            {
                "role": "user",
                "content": f"请分析以下文本的情感倾向：\n\n{text_block}",
            },
        ]

        try:
            return self.chat_json(messages)
        except Exception:
            return {
                "items": [],
                "overall": {
                    "positive_ratio": 0.33,
                    "negative_ratio": 0.33,
                    "neutral_ratio": 0.34,
                    "dominant_sentiment": "neutral",
                    "sentiment_summary": "情感分析因系统错误未能完成",
                },
            }

    def _extract_opinions(self, query: str, texts: list[dict], sentiment_data: dict) -> dict:
        """Extract key opinions and viewpoints from the analyzed content."""
        self.update_status("analyzing", "正在提取关键观点...", 0.6)

        text_block = "\n\n".join([
            f"[{i+1}] {t.get('title', '')}: {t.get('snippet', t.get('content', ''))}"
            for i, t in enumerate(texts[:15])
        ])

        messages = [
            {
                "role": "system",
                "content": (
                    "你是舆情分析专家。请从以下内容中提取关键观点，分析各方立场。"
                    "返回JSON格式：\n"
                    "{\n"
                    '  "main_viewpoints": [\n'
                    '    {"viewpoint": "观点描述", "stance": "支持/反对/中立", "representative_source": "代表性来源", "influence": "high/medium/low"}\n'
                    "  ],\n"
                    '  "controversy_points": ["争议点1", "争议点2"],\n'
                    '  "public_concerns": ["公众关注点1", "公众关注点2"],\n'
                    '  "opinion_leaders": ["意见领袖/关键信源1"],\n'
                    '  "emotional_triggers": ["情绪触发点1", "情绪触发点2"],\n'
                    '  "narrative_analysis": "舆论叙事分析（200字）"\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": f"分析主题: {query}\n\n内容:\n{text_block}",
            },
        ]

        try:
            return self.chat_json(messages)
        except Exception:
            return {
                "main_viewpoints": [],
                "controversy_points": [],
                "public_concerns": [],
                "opinion_leaders": [],
                "emotional_triggers": [],
                "narrative_analysis": "观点提取因系统错误未能完成",
            }

    def run(self, query: str, forum_context: list[dict] | None = None) -> dict:
        """Execute sentiment analysis workflow."""
        self.update_status("starting", f"开始情感分析: {query}", 0.0)

        try:
            # Gather texts to analyze from forum context (shared by QueryAgent)
            texts = []
            if forum_context:
                for post in forum_context:
                    if post.get("agent") == "QueryAgent":
                        # Use findings from QueryAgent
                        texts.extend(post.get("findings", []))

            # If no forum data, this agent can do its own search
            if not texts:
                self.update_status("searching", "未获取到共享数据，正在独立搜索...", 0.1)
                from tools.search import search_web, search_news
                web = search_web(query, max_results=10)
                news = search_news(query, max_results=10)
                texts = web + news

            if not texts:
                self.update_status("done", "未找到可分析的内容", 1.0)
                return {"error": "No content to analyze"}

            # Step 1: Sentiment analysis
            sentiment_data = self._analyze_sentiment_batch(texts)

            # Step 2: Opinion extraction
            opinion_data = self._extract_opinions(query, texts, sentiment_data)

            # Step 3: Generate keywords
            all_text = " ".join([t.get("snippet", t.get("content", "")) for t in texts])
            keywords = extract_keywords(all_text, top_n=15)

            # Step 4: Post to forum
            overall = sentiment_data.get("overall", {})
            forum_summary = (
                f"## SentimentAgent 情感分析报告\n\n"
                f"**整体情感倾向**: {overall.get('dominant_sentiment', 'N/A')}\n"
                f"- 正面: {overall.get('positive_ratio', 0):.0%}\n"
                f"- 负面: {overall.get('negative_ratio', 0):.0%}\n"
                f"- 中性: {overall.get('neutral_ratio', 0):.0%}\n\n"
                f"**情感摘要**: {overall.get('sentiment_summary', 'N/A')}\n\n"
                f"**关键词**: {', '.join(keywords[:10])}\n\n"
                f"**叙事分析**: {opinion_data.get('narrative_analysis', 'N/A')}"
            )
            self.post_to_forum(forum_summary, topic="情感分析结果")

            # Record findings
            self.add_finding({
                "type": "sentiment_analysis",
                "overall_sentiment": overall,
                "keywords": keywords,
            })

            self.update_status("done", "情感分析完成", 1.0)

            return {
                "sentiment": sentiment_data,
                "opinions": opinion_data,
                "keywords": keywords,
                "analyzed_count": len(texts),
            }

        except Exception as e:
            self.update_status("error", f"情感分析失败: {str(e)}")
            return {"error": str(e)}
