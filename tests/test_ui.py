import json
from pathlib import Path

import httpx
import pytest
from dash.development.base_component import Component

from ui.api.client import APIClient, APIUnavailable
from ui.app import create_app, render_page
from ui.components.navigation import NAV_ITEMS, navigation
from ui.config import UISettings


def walk(value):
    if isinstance(value, Component):
        yield value
        yield from walk(getattr(value, "children", None))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk(item)


def text_content(value):
    if isinstance(value, Component):
        return text_content(getattr(value, "children", None))
    if isinstance(value, (list, tuple)):
        return " ".join(text_content(item) for item in value)
    return str(value or "")


def empty_api():
    def handler(request):
        path = request.url.path
        if path.endswith("/health/live"):
            return httpx.Response(200, json={"status": "ok"})
        if path.endswith("/health/dependencies"):
            return httpx.Response(
                200,
                json={
                    "database": {
                        "configured": True,
                        "initialized": True,
                        "engine": "sqlite",
                    },
                    "embeddings": {
                        "configured": True,
                        "initialized": False,
                        "model": "GTE",
                    },
                    "vector_store": {
                        "configured": True,
                        "initialized": False,
                        "mode": "local",
                    },
                    "llm": {
                        "configured": False,
                        "initialized": False,
                        "model": "gpt-4o-mini",
                    },
                },
            )
        return httpx.Response(200, json=[])

    return APIClient(UISettings(_env_file=None), httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/", "Compliance workspace"),
        ("/policies", "No policies yet"),
        ("/check", "No check has been run"),
        ("/history", "No checks yet"),
    ],
)
def test_navigation_and_empty_states(path, expected):
    page, status = render_page(path, empty_api())
    assert expected in text_content(page)
    assert "Backend online" in text_content(status)
    active = [link for link in navigation(path) if "active" in link.className]
    assert len(active) == 1
    assert active[0].href == path


@pytest.mark.parametrize("path", ["/policies", "/check"])
def test_unfinished_workflow_controls_are_disabled(path):
    page, _ = render_page(path, empty_api())
    controls = [
        node for node in walk(page) if type(node).__name__ in {"Button", "Upload", "Dropdown"}
    ]
    assert controls
    assert all(node.disabled is True for node in controls)
    assert "not implemented yet" in text_content(page)


def test_api_unavailable_feedback_and_unknown_page():
    def offline(request):
        raise httpx.ConnectError("offline", request=request)

    client = APIClient(UISettings(_env_file=None), httpx.MockTransport(offline))
    page, status = render_page("/", client)
    assert "Start the API" in text_content(page)
    assert page.role == "alert"
    assert "Backend unavailable" in text_content(status)
    page, _ = render_page("/unknown", client)
    assert "Page not found" in text_content(page)


@pytest.mark.parametrize("failure", ["timeout", "http", "invalid_json", "invalid_list"])
def test_api_client_error_handling(failure):
    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "http":
            return httpx.Response(500)
        if failure == "invalid_json":
            return httpx.Response(200, text="invalid")
        return httpx.Response(200, json={"unexpected": "object"})

    client = APIClient(UISettings(_env_file=None), httpx.MockTransport(handler))
    with pytest.raises(APIUnavailable):
        client.documents()


def test_dash_routes_assets_and_real_callback():
    app = create_app(UISettings(_env_file=None), empty_api())
    browser = app.server.test_client()
    for path, _, _ in NAV_ITEMS:
        assert browser.get(path).status_code == 200
    assert browser.get("/_dash-layout").status_code == 200
    assert browser.get("/_dash-dependencies").status_code == 200
    assert browser.get("/assets/styles.css").status_code == 200
    output = next(iter(app.callback_map))
    response = browser.post(
        "/_dash-update-component",
        json={
            "output": output,
            "outputs": [
                {"id": "page-content", "property": "children"},
                {"id": "api-status", "property": "children"},
                {"id": "navigation", "property": "children"},
            ],
            "inputs": [
                {"id": "url", "property": "pathname", "value": "/policies"},
                {"id": "refresh-button", "property": "n_clicks", "value": 0},
                {"id": "refresh-interval", "property": "n_intervals", "value": 0},
            ],
            "state": [],
            "changedPropIds": ["url.pathname"],
        },
    )
    assert response.status_code == 200
    assert "No policies yet" in json.dumps(response.get_json())


def test_ui_is_independent_of_backend_and_secrets():
    root = Path(__file__).resolve().parents[1] / "ui"
    for path in root.rglob("*.py"):
        source = path.read_text()
        assert "from backend" not in source
        assert "import backend" not in source
    assert "openai_api_key" not in UISettings.model_fields
