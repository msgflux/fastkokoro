from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from fastkokoro.audio import media_type
from fastkokoro.config import Settings
from fastkokoro.engine import FastKokoro
from fastkokoro.json import FastJSONResponse
from fastkokoro.metrics import Metrics
from fastkokoro.openai import ModelList, ModelObject, SpeechRequest
from fastkokoro.profiling import Profiler
from fastkokoro.voices import KOKORO_MODEL_ID, SUPPORTED_MODEL_IDS


def iter_warmup_request_texts(settings: Settings) -> Iterator[str]:
    seen: set[str] = set()

    def emit(text: str) -> Iterator[str]:
        candidate = text.strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        yield candidate

    yield from emit(settings.warmup_text)


def create_app(
    engine: FastKokoro | None = None,
    settings: Settings | None = None,
    metrics: Metrics | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app_engine = get_engine()
        app_profiler = get_profiler()
        settings = get_settings()
        if hasattr(app_engine, "settings") and app_engine.settings.warmup:
            with app_profiler.capture(
                "startup-warmup",
                enabled=app_profiler.profile_warmup,
            ):
                app_engine.warmup()
        if settings.warmup_request:
            with app_profiler.capture(
                "startup-warmup-request",
                enabled=app_profiler.profile_warmup,
            ):
                await warmup_request()
        yield

    app = FastAPI(
        title="fastkokoro",
        version="0.1.0",
        default_response_class=FastJSONResponse,
        lifespan=lifespan,
    )
    app.state.engine = engine
    app.state.settings = settings
    app.state.metrics = metrics or Metrics()
    app.state.profiler = None

    def get_settings() -> Settings:
        if app.state.settings is not None:
            return app.state.settings
        if app.state.engine is not None and hasattr(app.state.engine, "settings"):
            app.state.settings = app.state.engine.settings
        else:
            app.state.settings = Settings.from_env()
        return app.state.settings

    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_methods=list(settings.cors_allow_methods),
        allow_headers=list(settings.cors_allow_headers),
        allow_credentials=settings.cors_allow_credentials,
    )

    def get_engine() -> FastKokoro:
        if app.state.engine is None:
            app.state.engine = FastKokoro(get_settings())
        return app.state.engine

    def get_profiler() -> Profiler:
        if app.state.profiler is None:
            app.state.profiler = Profiler(get_settings())
        return app.state.profiler


    async def generate_speech_response(
        request: SpeechRequest,
        *,
        record_metrics: bool = True,
    ) -> Response:
        start = time.perf_counter()
        profiler = get_profiler()
        if request.model not in SUPPORTED_MODEL_IDS:
            if record_metrics:
                app.state.metrics.record_speech(
                    streaming=False,
                    latency_seconds=time.perf_counter() - start,
                    error=True,
                )
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
            if record_metrics:
                app.state.metrics.record_speech(
                    streaming=should_stream,
                    latency_seconds=time.perf_counter() - start,
                    error=True,
                )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if should_stream:

            async def chunks() -> AsyncGenerator[bytes, None]:
                chunk_count = 0
                bytes_count = 0
                first_chunk_latency = None
                error = False
                try:
                    with profiler.capture(
                        "speech-stream",
                        enabled=profiler.profile_requests and record_metrics,
                    ):
                        async for chunk in engine.create_stream(
                            request.input,
                            voice=resolved_voice,
                            speed=request.speed,
                            response_format=request.response_format,
                            lang=resolved_lang,
                        ):
                            if first_chunk_latency is None:
                                first_chunk_latency = time.perf_counter() - start
                            chunk_count += 1
                            bytes_count += len(chunk)
                            yield chunk
                except (AssertionError, RuntimeError, ValueError):
                    error = True
                    raise
                finally:
                    if record_metrics:
                        app.state.metrics.record_speech(
                            streaming=True,
                            latency_seconds=time.perf_counter() - start,
                            first_chunk_latency_seconds=first_chunk_latency,
                            chunks=chunk_count,
                            bytes_count=bytes_count,
                            error=error,
                        )

            return StreamingResponse(chunks(), media_type=content_type)

        try:
            with profiler.capture(
                "speech-non-streaming",
                enabled=profiler.profile_requests and record_metrics,
            ):
                audio = engine.create(
                    request.input,
                    voice=resolved_voice,
                    speed=request.speed,
                    response_format=request.response_format,
                    lang=resolved_lang,
                )
        except (AssertionError, RuntimeError, ValueError) as exc:
            if record_metrics:
                app.state.metrics.record_speech(
                    streaming=False,
                    latency_seconds=time.perf_counter() - start,
                    error=True,
                )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if record_metrics:
            app.state.metrics.record_speech(
                streaming=False,
                latency_seconds=time.perf_counter() - start,
                bytes_count=len(audio),
            )
        return Response(content=audio, media_type=content_type)

    async def warmup_request() -> None:
        settings = get_settings()
        for text in iter_warmup_request_texts(settings):
            response = await generate_speech_response(
                SpeechRequest(
                    input=text,
                    voice=settings.default_voice,
                    response_format="pcm",
                    stream=True,
                    lang=settings.default_lang,
                ),
                record_metrics=False,
            )
            if not isinstance(response, StreamingResponse):
                continue
            iterator = response.body_iterator
            try:
                async for _ in iterator:
                    break
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()

    @app.middleware("http")
    async def collect_http_metrics(request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            app.state.metrics.record_request(
                request.url.path,
                status_code,
                time.perf_counter() - start,
            )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/metrics")
    def metrics() -> dict:
        snapshot = app.state.metrics.snapshot()
        engine = get_engine()
        snapshot["runtime"] = {
            "model_repo": engine.settings.model_repo,
            "model_file": engine.settings.model_file,
            "active_providers": engine.session.get_providers(),
            "configured_providers": list(engine.settings.onnx_providers),
            "onnx_auto_providers": engine.settings.onnx_auto_providers,
            "profiling": get_profiler().snapshot(),
        }
        return snapshot

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
        return await generate_speech_response(request)

    return app


app = create_app()
