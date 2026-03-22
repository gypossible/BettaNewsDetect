"""Flask main application - Entry point for the public opinion analysis system.

Provides:
- Web interface for submitting analysis queries
- SSE streaming for real-time progress updates
- REST API for reports management
- Multi-agent orchestration (QueryAgent + SentimentAgent → Forum → ReportAgent)
"""

import os
import sys
import json
import time
import uuid
import logging
import threading
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from agents.query_agent import QueryAgent
from agents.sentiment_agent import SentimentAgent
from agents.report_agent import ReportAgent
from forum.engine import ForumEngine
from forum.moderator import ForumModerator
from report.renderer import render_html_report, save_report
from report.generator import prepare_report_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Task storage (in-memory)
tasks: dict[str, dict] = {}


def run_analysis(task_id: str, query: str):
    """Run the full multi-agent analysis pipeline in a background thread."""
    task = tasks[task_id]
    task["status"] = "running"
    task["events"] = []

    def add_event(event_type: str, data: dict):
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        task["events"].append(event)

    try:
        add_event("system", {"message": "🚀 分析系统启动...", "phase": "init"})

        # Validate config
        warnings = Config.validate()
        if warnings:
            for w in warnings:
                add_event("warning", {"message": f"⚠️ {w}"})
            if not Config.LLM_API_KEY:
                add_event("error", {"message": "❌ LLM API Key 未配置，无法进行分析。请在 .env 文件中设置 LLM_API_KEY。"})
                task["status"] = "error"
                task["error"] = "LLM API Key not configured"
                return

        # Initialize agents and forum
        query_agent = QueryAgent()
        sentiment_agent = SentimentAgent()
        report_agent = ReportAgent()
        forum = ForumEngine()
        moderator = ForumModerator()

        add_event("system", {"message": "🤖 智能体已就绪: QueryAgent, SentimentAgent, ReportAgent", "phase": "agents_ready"})

        # ===== Phase 1: Parallel Search & Data Collection =====
        add_event("phase", {"phase": "search", "message": "📡 Phase 1: 信息搜索与收集"})

        # Run QueryAgent
        add_event("agent_start", {"agent": "QueryAgent", "message": "🔍 QueryAgent 开始搜索..."})
        query_results = query_agent.run(query)

        # Drain events from agent
        for evt in query_agent.get_events():
            add_event(evt["type"], evt["data"])

        # Post QueryAgent findings to forum
        for post in query_agent.state.forum_posts:
            forum.post(post["agent"], post.get("role", ""), post["content"], post.get("topic", ""))

        add_event("agent_done", {"agent": "QueryAgent", "message": f"✅ QueryAgent 完成，找到 {query_results.get('total_sources', 0)} 条来源"})

        # ===== Phase 2: Sentiment Analysis =====
        add_event("phase", {"phase": "sentiment", "message": "💭 Phase 2: 情感分析与观点挖掘"})

        # Pass QueryAgent findings to SentimentAgent
        sentiment_context = [{
            "agent": "QueryAgent",
            "findings": query_agent.state.findings,
        }]

        add_event("agent_start", {"agent": "SentimentAgent", "message": "🎭 SentimentAgent 开始分析..."})
        sentiment_results = sentiment_agent.run(query, forum_context=sentiment_context)

        for evt in sentiment_agent.get_events():
            add_event(evt["type"], evt["data"])

        for post in sentiment_agent.state.forum_posts:
            forum.post(post["agent"], post.get("role", ""), post["content"], post.get("topic", ""))

        add_event("agent_done", {"agent": "SentimentAgent", "message": "✅ SentimentAgent 分析完成"})

        # ===== Phase 3: Forum Discussion =====
        add_event("phase", {"phase": "forum", "message": "🗣️ Phase 3: 论坛协作讨论"})

        forum.start_new_round()
        moderation = moderator.moderate(query, forum.get_all_posts(), 1)
        add_event("forum", {
            "round": 1,
            "moderator_comment": moderation.get("moderator_comment", ""),
            "is_sufficient": moderation.get("is_sufficient", True),
        })

        # ===== Phase 4: Report Generation =====
        add_event("phase", {"phase": "report", "message": "📝 Phase 4: 报告生成"})

        all_results = {
            "query": query_results,
            "sentiment": sentiment_results,
            "forum_posts": forum.get_all_posts(),
        }

        add_event("agent_start", {"agent": "ReportAgent", "message": "📊 ReportAgent 开始生成报告..."})
        report_data = report_agent.run(query, all_results=all_results)

        for evt in report_agent.get_events():
            add_event(evt["type"], evt["data"])

        # Enrich report_data with auxiliary data
        report_data["sentiment_data"] = sentiment_results.get("sentiment", {}).get("overall", {})
        report_data["keywords"] = sentiment_results.get("keywords", [])
        report_data["sources"] = []
        web_sources = query_results.get("sources", {}).get("web", [])
        news_sources = query_results.get("sources", {}).get("news", [])
        for s in (web_sources + news_sources)[:15]:
            report_data["sources"].append({
                "title": s.get("title", ""),
                "url": s.get("url", ""),
                "type": s.get("source", "web"),
            })
        report_data["total_sources"] = query_results.get("total_sources", 0)
        report_data["analyzed_count"] = sentiment_results.get("analyzed_count", 0)

        # ===== Phase 5: Render & Save Report =====
        add_event("phase", {"phase": "render", "message": "🎨 Phase 5: 报告渲染"})

        html_content = render_html_report(report_data)
        report_filename = f"report_{task_id[:8]}_{time.strftime('%Y%m%d_%H%M%S')}.html"
        report_path = save_report(html_content, report_filename)

        add_event("agent_done", {"agent": "ReportAgent", "message": "✅ 报告生成完成"})
        add_event("complete", {
            "message": "🎉 分析完成！",
            "report_filename": report_filename,
            "report_url": f"/reports/{report_filename}",
        })

        task["status"] = "done"
        task["result"] = {
            "report_filename": report_filename,
            "report_url": f"/reports/{report_filename}",
            "summary": report_data.get("executive_summary", ""),
        }

    except Exception as e:
        logger.exception(f"Analysis failed for task {task_id}")
        add_event("error", {"message": f"❌ 分析失败: {str(e)}"})
        task["status"] = "error"
        task["error"] = str(e)


# ===== Routes =====

@app.route("/")
def index():
    """Serve the main web page."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def start_analysis():
    """Start a new analysis task."""
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "请输入分析主题"}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "id": task_id,
        "query": query,
        "status": "pending",
        "created_at": time.time(),
        "events": [],
        "result": None,
        "error": None,
    }

    # Start analysis in background thread
    thread = threading.Thread(target=run_analysis, args=(task_id, query), daemon=True)
    thread.start()

    return jsonify({"task_id": task_id, "status": "pending"})


@app.route("/api/stream/<task_id>")
def stream_events(task_id: str):
    """SSE endpoint for real-time progress updates."""
    if task_id not in tasks:
        return jsonify({"error": "Task not found"}), 404

    def generate():
        last_index = 0
        while True:
            task = tasks.get(task_id)
            if not task:
                break

            events = task["events"]
            while last_index < len(events):
                event = events[last_index]
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                last_index += 1

            if task["status"] in ("done", "error"):
                # Send final status
                final = {
                    "type": "final",
                    "data": {
                        "status": task["status"],
                        "result": task.get("result"),
                        "error": task.get("error"),
                    },
                }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                break

            time.sleep(0.5)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/reports")
def list_reports():
    """List all generated reports."""
    report_dir = Config.REPORT_OUTPUT_DIR
    if not os.path.exists(report_dir):
        return jsonify({"reports": []})

    reports = []
    for f in sorted(os.listdir(report_dir), reverse=True):
        if f.endswith(".html"):
            filepath = os.path.join(report_dir, f)
            reports.append({
                "filename": f,
                "url": f"/reports/{f}",
                "size": os.path.getsize(filepath),
                "created": os.path.getmtime(filepath),
            })

    return jsonify({"reports": reports})


@app.route("/reports/<filename>")
def serve_report(filename):
    """Serve a generated report file."""
    return send_from_directory(Config.REPORT_OUTPUT_DIR, filename)


if __name__ == "__main__":
    # Validate configuration
    warnings = Config.validate()
    for w in warnings:
        logger.warning(w)

    print("\n" + "=" * 60)
    print("  🐟 微舆分析系统 (BettaFish-Lite) 启动中...")
    print("=" * 60)
    print(f"  🌐 访问地址: http://localhost:{Config.FLASK_PORT}")
    print(f"  🤖 LLM模型: {Config.LLM_MODEL}")
    print(f"  🔑 API Key: {'已配置' if Config.LLM_API_KEY else '❌ 未配置'}")
    print("=" * 60 + "\n")

    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG,
    )
