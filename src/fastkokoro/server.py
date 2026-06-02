from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from fastkokoro.audio import media_type
from fastkokoro.engine import FastKokoro
from fastkokoro.json import FastJSONResponse
from fastkokoro.openai import ModelList, ModelObject, SpeechRequest
from fastkokoro.voices import KOKORO_MODEL_ID, SUPPORTED_MODEL_IDS


def create_app(engine: FastKokoro | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_engine = get_engine()
        if app_engine.settings.warmup:
            app_engine.warmup()
        yield

    app = FastAPI(
        title="fastkokoro",
        version="0.1.0",
        default_response_class=FastJSONResponse,
        lifespan=lifespan,
    )
    app.state.engine = engine

    def get_engine() -> FastKokoro:
        if app.state.engine is None:
            app.state.engine = FastKokoro()
        return app.state.engine

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/v1/models")
    def models() -> ModelList:
        return ModelList(
            data=[
                ModelObject(id=KOKORO_MODEL_ID),
            ]
        )

    @app.get("/v1/audio/voices")
    def voices() -> dict[str, list[str]]:
        return {"voices": get_engine().voices()}

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        if request.model not in SUPPORTED_MODEL_IDS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model {request.model!r}. Use {KOKORO_MODEL_ID!r}.",
            )

        engine = get_engine()
        content_type = media_type(request.response_format)
        should_stream = request.stream is True

        try:
            resolved_voice, resolved_lang = engine.resolve_request(
                request.voice, request.lang
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if should_stream:

            async def chunks() -> AsyncGenerator[bytes, None]:
                async for chunk in engine.create_stream(
                    request.input,
                    voice=resolved_voice,
                    speed=request.speed,
                    response_format=request.response_format,
                    lang=resolved_lang,
                ):
                    yield chunk

            return StreamingResponse(chunks(), media_type=content_type)

        try:
            audio = engine.create(
                request.input,
                voice=resolved_voice,
                speed=request.speed,
                response_format=request.response_format,
                lang=resolved_lang,
            )
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=audio, media_type=content_type)

    return app


app = create_app()
