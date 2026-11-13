"""Serve a local Viser viewer through a Gradio application's HTTP port."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect

import httpx
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake


class ViserProxy:
    """Proxy viewer assets and binary WebSocket messages to the active viewer."""

    def __init__(self, app: ASGIApp, get_port: Callable[[], int | None],
                 prefix: str = "/stage2-viewer"):
        self.app = app
        self.get_port = get_port
        self.prefix = prefix.rstrip("/")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        root = scope.get("root_path", "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root):]
        if path != self.prefix and not path.startswith(self.prefix + "/"):
            await self.app(scope, receive, send)
            return
        port = self.get_port()
        if port is None:
            if scope["type"] == "websocket":
                await WebSocket(scope, receive, send).close(code=1013)
            else:
                await HTMLResponse("Run inference to load the trajectory viewer.",
                                   status_code=503)(scope, receive, send)
            return
        if scope["type"] == "http" and path == self.prefix:
            await RedirectResponse(root + self.prefix + "/")(scope, receive, send)
            return
        upstream_path = path[len(self.prefix):] or "/"
        query = scope.get("query_string", b"").decode("ascii")
        if query:
            upstream_path += "?" + query
        if scope["type"] == "websocket":
            await self._websocket(scope, receive, send, port, upstream_path)
            return
        request = Request(scope, receive)
        if request.method not in {"GET", "HEAD"}:
            await Response(status_code=405, headers={"Allow": "GET, HEAD"})(scope, receive, send)
            return
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                upstream = await client.get(
                    f"http://127.0.0.1:{port}{upstream_path}",
                    headers={"Accept-Encoding": "identity"})
            # httpx decodes HTTP content encodings; let the UI server encode
            # the response again rather than forwarding stale lengths/encodings.
            headers = {key: value for key, value in upstream.headers.items()
                       if key.lower() not in {
                           "connection", "content-encoding", "content-length",
                           "keep-alive", "transfer-encoding", "upgrade"}}
            headers["content-length"] = str(len(upstream.content))
            response = Response(upstream.content if request.method == "GET" else b"",
                                status_code=upstream.status_code, headers=headers)
        except httpx.RequestError:
            response = HTMLResponse("The viewer is restarting. Please retry shortly.", status_code=503)
        await response(scope, receive, send)

    async def _websocket(self, scope: Scope, receive: Receive, send: Send,
                         port: int, path: str) -> None:
        browser = WebSocket(scope, receive, send)
        options = {}
        # websockets 15+ can use system proxies. The upstream is always a
        # private loopback service; older supported versions have no option.
        if "proxy" in inspect.signature(connect).parameters:
            options["proxy"] = None
        tasks = []
        try:
            async with connect(f"ws://127.0.0.1:{port}{path}",
                               subprotocols=scope.get("subprotocols") or None,
                               max_size=50 * 1024**2, open_timeout=5, **options) as upstream:
                await browser.accept(subprotocol=upstream.subprotocol)

                async def to_viewer():
                    while True:
                        message = await browser.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        if message.get("bytes") is not None:
                            await upstream.send(message["bytes"])
                        elif message.get("text") is not None:
                            await upstream.send(message["text"])

                async def to_browser():
                    async for message in upstream:
                        if isinstance(message, bytes):
                            await browser.send_bytes(message)
                        else:
                            await browser.send_text(message)

                tasks = [asyncio.create_task(to_viewer()), asyncio.create_task(to_browser())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
        except (OSError, TimeoutError, ConnectionClosed, InvalidHandshake, WebSocketDisconnect):
            pass
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if (browser.application_state == WebSocketState.CONNECTED
                    and browser.client_state == WebSocketState.CONNECTED):
                await browser.close()
            elif browser.application_state == WebSocketState.CONNECTING:
                await browser.close(code=1013)
