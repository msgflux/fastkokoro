from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse

from fastkokoro.audio import media_type
from fastkokoro.engine import FastKokoro
from fastkokoro.json import FastJSONResponse
from fastkokoro.openai import ModelList, ModelObject, SpeechRequest


def create_app(engine: FastKokoro | None = None) -> FastAPI:
    app = FastAPI(
        title="fastkokoro",
        version="0.1.0",
        default_response_class=FastJSONResponse,
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
                ModelObject(id="kokoro"),
                ModelObject(id="tts-1"),
                ModelObject(id="gpt-4o-mini-tts"),
            ]
        )

    @app.get("/v1/audio/voices")
    def voices() -> dict[str, list[str]]:
        return {"voices": get_engine().voices()}

    @app.post("/v1/audio/speech")
    async def speech(request: SpeechRequest) -> Response:
        engine = get_engine()
        content_type = media_type(request.response_format)
        should_stream = request.stream is True

        if should_stream:

            async def chunks() -> AsyncGenerator[bytes, None]:
                async for chunk in engine.create_stream(
                    request.input,
                    voice=request.voice,
                    speed=request.speed,
                    response_format=request.response_format,
                    lang=request.lang,
                ):
                    yield chunk

            return StreamingResponse(chunks(), media_type=content_type)

        audio = engine.create(
            request.input,
            voice=request.voice,
            speed=request.speed,
            response_format=request.response_format,
            lang=request.lang,
        )
        return Response(content=audio, media_type=content_type)

    return app


app = create_app()
