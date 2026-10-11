import base64
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
def test_workflow_controls_are_enabled(path):
    page, _ = render_page(path, empty_api())
    controls = [
        node for node in walk(page) if type(node).__name__ in {"Button", "Upload", "Dropdown"}
    ]
    assert controls
    assert all(getattr(node, "disabled", False) is not True for node in controls)
    assert "not implemented yet" not in text_content(page)


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
                {"id": "workflow-version", "property": "data", "value": None},
                {"id": "last-check", "property": "data", "value": None},
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


def test_upload_callback_sends_multipart_document_and_refreshes_library():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            201, json={"id": "document", "filename": "policy.csv", "status": "ready"}
        )

    client = APIClient(UISettings(_env_file=None), httpx.MockTransport(handler))
    app = create_app(UISettings(_env_file=None), client)
    output = next(
        key
        for key, value in app.callback_map.items()
        if value["inputs"][0]["id"] == "policy-upload"
    )
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output,
            "outputs": [
                {"id": "workflow-status", "property": "children"},
                {"id": "workflow-version", "property": "data"},
            ],
            "inputs": [
                {
                    "id": "policy-upload",
                    "property": "contents",
                    "value": "data:text/csv;base64,"
                    + base64.b64encode(b"Name,Value\nStrength,100\n").decode(),
                }
            ],
            "state": [{"id": "policy-upload", "property": "filename", "value": "policy.csv"}],
            "changedPropIds": ["policy-upload.contents"],
        },
    )
    assert response.status_code == 200
    assert "indexed and ready" in json.dumps(response.json)
    assert requests[0].url.path.endswith("/documents")
    assert b"Strength,100" in requests[0].content
    assert b"policy" in requests[0].content


def test_check_callback_submits_specifications_and_persists_result():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            201, json={"id": "check", "status": "completed", "result": {"summary": "Done"}}
        )

    client = APIClient(UISettings(_env_file=None), httpx.MockTransport(handler))
    app = create_app(UISettings(_env_file=None), client)
    output = next(
        key for key, value in app.callback_map.items() if value["inputs"][0]["id"] == "run-check"
    )
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": output,
            "outputs": [
                {"id": "last-check", "property": "data"},
                {"id": "workflow-status", "property": "children"},
            ],
            "inputs": [{"id": "run-check", "property": "n_clicks", "value": 1}],
            "state": [
                {"id": "subject-selection", "property": "value", "value": "supplier"},
                {"id": "policy-selection", "property": "value", "value": ["policy"]},
                {
                    "id": "specifications",
                    "property": "value",
                    "value": " Strength >= 100 MPa \n\n Certificate required",
                },
            ],
            "changedPropIds": ["run-check.n_clicks"],
        },
    )
    assert response.status_code == 200
    assert response.json["response"]["last-check"]["data"]["id"] == "check"
    assert json.loads(requests[0].content)["specifications"] == [
        "Strength >= 100 MPa",
        "Certificate required",
    ]


def test_completed_results_show_download_options_and_confidence():
    check = {
        "id": "review",
        "result": {
            "verdict": "needs_review",
            "summary": "Review needed",
            "comparison_matrix": [
                {
                    "specification": "Strength",
                    "satisfaction": "partial",
                    "extent_percent": 50,
                    "confidence": 0.8,
                    "explanation": "Evidence incomplete",
                }
            ],
            "clarification_markdown": "# Supplier clarification sheet",
        },
    }
    page, _ = render_page("/check", empty_api(), check)
    content = text_content(page)
    assert "50%" in content and "80%" in content
    links = [node for node in walk(page) if type(node).__name__ == "A"]
    assert {node.children for node in links} >= {"Download MD", "Download DOCX", "Download PDF"}
