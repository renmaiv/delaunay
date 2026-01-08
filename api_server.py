"""
FastAPI Server for Conversation Safety Judge

Provides REST API endpoints for jailbreak detection.
"""

from typing import List, Dict, Optional
from datetime import datetime
import uvicorn

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not installed. Install with: pip install fastapi uvicorn")
    FASTAPI_AVAILABLE = False

from config_loader import load_config
from data_loader import Conversation
from bert_classifier import BERTClassifier
from conversation_judge import ConversationJudge, CaseMaterial


# Request/Response Models
class Message(BaseModel):
    """A message in a conversation"""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class JudgeRequest(BaseModel):
    """Request to judge a conversation"""
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID")
    messages: List[Message] = Field(..., description="List of messages")


class JudgeResponse(BaseModel):
    """Response from judge"""
    conversation_id: str
    threat_type: str
    confidence: float
    risk_score: float
    reasoning: str
    timestamp: str


class BatchJudgeRequest(BaseModel):
    """Batch judge request"""
    conversations: List[JudgeRequest]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    judge_mode: str
    version: str


# Initialize FastAPI app
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Conversation Safety Judge API",
        description="API for detecting jailbreaks and conversation manipulation",
        version="1.0.0"
    )

    # Global state
    config = None
    classifier = None
    judge = None


    def get_classifier():
        """Get or initialize classifier"""
        global classifier, judge, config

        if config is None:
            config = load_config()

        judge_mode = config.get_judge_mode()

        if judge_mode == "bert":
            if classifier is None:
                bert_config = config.get_bert_config()
                classifier = BERTClassifier(
                    model_name=bert_config.get("model_name", "bert-base-uncased"),
                    device=bert_config.get("device", "cpu")
                )
            return classifier, None
        else:
            if judge is None:
                llm_config = config.get_llm_config()
                judge = ConversationJudge(
                    llm_provider=llm_config.get("provider", "mock"),
                    model=llm_config.get("model", "gpt-4"),
                    api_key=llm_config.get("api_key")
                )
            return None, judge


    @app.get("/", response_model=HealthResponse)
    async def root():
        """Root endpoint"""
        global config
        if config is None:
            config = load_config()

        return HealthResponse(
            status="healthy",
            judge_mode=config.get_judge_mode(),
            version="1.0.0"
        )


    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        global config
        if config is None:
            config = load_config()

        return HealthResponse(
            status="healthy",
            judge_mode=config.get_judge_mode(),
            version="1.0.0"
        )


    @app.post("/judge", response_model=JudgeResponse)
    async def judge_conversation(request: JudgeRequest):
        """
        Judge a single conversation

        Args:
            request: JudgeRequest with messages

        Returns:
            JudgeResponse with verdict
        """
        try:
            classifier, judge = get_classifier()

            # Convert to internal format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

            conversation_id = request.conversation_id or f"api_{datetime.now().timestamp()}"

            if classifier is not None:
                # BERT mode
                result = classifier.classify_conversation(messages)

                response = JudgeResponse(
                    conversation_id=conversation_id,
                    threat_type=result.threat_type,
                    confidence=result.confidence,
                    risk_score=result.confidence * 100,
                    reasoning=result.reasoning,
                    timestamp=datetime.utcnow().isoformat()
                )

            else:
                # LLM mode
                case = CaseMaterial(
                    case_id=conversation_id,
                    messages=messages
                )

                verdict = judge.judge_conversation(case)

                response = JudgeResponse(
                    conversation_id=conversation_id,
                    threat_type=verdict.threat_type.value,
                    confidence=verdict.confidence,
                    risk_score=verdict.risk_score,
                    reasoning=verdict.reasoning,
                    timestamp=verdict.timestamp
                )

            return response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.post("/batch")
    async def batch_judge(request: BatchJudgeRequest, background_tasks: BackgroundTasks):
        """
        Batch judge multiple conversations

        Args:
            request: BatchJudgeRequest with multiple conversations

        Returns:
            List of JudgeResponse objects
        """
        try:
            results = []

            for conv_request in request.conversations:
                result = await judge_conversation(conv_request)
                results.append(result)

            return {
                "total": len(results),
                "results": results
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.get("/config")
    async def get_config_info():
        """Get configuration information"""
        global config
        if config is None:
            config = load_config()

        return {
            "judge_mode": config.get_judge_mode(),
            "bert_model": config.get("bert.model_name"),
            "llm_provider": config.get("llm.provider")
        }


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False
):
    """
    Run the API server

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload (development only)
    """
    if not FASTAPI_AVAILABLE:
        print("Error: FastAPI not installed")
        print("Install with: pip install fastapi uvicorn")
        return

    print("=" * 70)
    print("CONVERSATION SAFETY JUDGE API SERVER")
    print("=" * 70)
    print(f"\nStarting server on http://{host}:{port}")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 70)

    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=reload
    )


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Run API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")

    args = parser.parse_args()

    # Load config
    load_config(args.config)

    # Run server
    run_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
