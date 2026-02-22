"""Entry point - starts both gRPC + FastAPI servers."""

import asyncio
import signal

import structlog
import uvicorn

from howler_agents_service.rest.app import create_app
from howler_agents_service.settings import settings

logger = structlog.get_logger()


async def main() -> None:
    """Start REST server. gRPC can be added alongside when proto stubs are generated."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(structlog, settings.log_level.upper(), 20)
        ),
    )

    app = create_app()
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=settings.rest_port,
        log_level=settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    logger.info("starting_service", rest_port=settings.rest_port, grpc_port=settings.grpc_port)

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, server.should_exit.__bool__)

    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
