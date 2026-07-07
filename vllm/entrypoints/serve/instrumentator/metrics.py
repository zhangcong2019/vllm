# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import prometheus_client
import regex as re
from fastapi import FastAPI, Response
from prometheus_client import make_asgi_app
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import routing as instrumentator_routing
from starlette.routing import Match
from starlette.routing import Mount

from vllm.v1.metrics.prometheus import get_prometheus_registry


class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST


def _patch_instrumentator_route_lookup():
    patch_marker = "_vllm_handles_pathless_routes"
    if getattr(instrumentator_routing._get_route_name, patch_marker, False):
        return

    def _get_route_name(scope, routes, route_name=None):
        for route in routes:
            match, child_scope = route.matches(scope)
            matched_route = route
            route_path = getattr(route, "path", None)

            if route_path is None and hasattr(route, "_match"):
                detailed_match, detailed_child_scope, detailed_route, _ = route._match(
                    scope
                )
                if detailed_match == match and detailed_route is not None:
                    child_scope = detailed_child_scope
                    matched_route = detailed_route
                    route_path = getattr(matched_route, "path", None)

            if route_path is None:
                if match == Match.FULL:
                    return route_name
                continue

            if match == Match.FULL:
                route_name = route_path
                child_scope = {**scope, **child_scope}
                if isinstance(matched_route, Mount) and matched_route.routes:
                    child_route_name = _get_route_name(
                        child_scope, matched_route.routes, route_name
                    )
                    if child_route_name is None:
                        route_name = None
                    else:
                        route_name += child_route_name
                return route_name
            elif match == Match.PARTIAL and route_name is None:
                route_name = route_path
        return None

    setattr(_get_route_name, patch_marker, True)
    instrumentator_routing._get_route_name = _get_route_name


def attach_router(app: FastAPI):
    """Mount prometheus metrics to a FastAPI app."""

    registry = get_prometheus_registry()
    _patch_instrumentator_route_lookup()

    # `response_class=PrometheusResponse` is needed to return an HTTP response
    # with header "Content-Type: text/plain; version=0.0.4; charset=utf-8"
    # instead of the default "application/json" which is incorrect.
    # See https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/163#issue-1296092364
    Instrumentator(
        excluded_handlers=[
            "/metrics",
            "/health",
            "/load",
            "/ping",
            "/version",
            "/server_info",
        ],
        registry=registry,
    ).add().instrument(app).expose(app, response_class=PrometheusResponse)

    # Add prometheus asgi middleware to route /metrics requests
    metrics_route = Mount("/metrics", make_asgi_app(registry=registry))

    # Workaround for 307 Redirect for /metrics
    metrics_route.path_regex = re.compile("^/metrics(?P<path>.*)$")
    app.routes.append(metrics_route)
