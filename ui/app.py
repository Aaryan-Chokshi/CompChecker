import base64
import binascii
from pathlib import Path
from uuid import uuid4

from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

from ui.api.client import APIClient, APIUnavailable
from ui.components.common import badge, empty_state, unavailable_state
from ui.components.navigation import NAV_ITEMS, navigation
from ui.config import UISettings
from ui.features.compliance.layout import check_page, history_page
from ui.features.documents.layout import policy_page
from ui.features.overview import overview_page


def render_page(pathname: str, client: APIClient, last_check=None):
    if pathname not in {item[0] for item in NAV_ITEMS}:
        return empty_state("Page not found", "Choose a view from the navigation."), badge(
            "Unknown page"
        )
    try:
        client.liveness()
        if pathname == "/":
            page = overview_page(client.documents(), client.checks(), client.dependencies())
        elif pathname == "/policies":
            page = policy_page(client.documents(role="policy"))
        elif pathname == "/check":
            page = check_page(
                client.documents(role="subject"),
                client.documents(role="policy"),
                last_check,
                client,
            )
        else:
            page = history_page(client.checks(), client)
        return page, badge("Backend online", "success")
    except APIUnavailable as exc:
        return unavailable_state(str(exc)), badge("Backend unavailable", "warning")


def create_app(settings: UISettings | None = None, api_client: APIClient | None = None) -> Dash:
    settings = settings or UISettings()
    client = api_client or APIClient(settings)
    app = Dash(
        __name__,
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
        title="Compliance Checker",
        update_title="Loading workspace…",
        suppress_callback_exceptions=True,
    )
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="workflow-version"),
            dcc.Store(id="last-check"),
            dcc.Interval(id="refresh-interval", interval=30_000, n_intervals=0),
            html.A("Skip to content", href="#page-content", className="skip-link"),
            html.Aside(
                [
                    html.Div(
                        [
                            html.Span("C", className="brand-mark"),
                            html.Div(
                                [
                                    html.Strong("Compliance"),
                                    html.Span("Workspace", className="brand-subtitle"),
                                ]
                            ),
                        ],
                        className="brand",
                    ),
                    html.P("WORKSPACE", className="eyebrow"),
                    html.Nav(
                        id="navigation",
                        children=navigation("/"),
                        **{"aria-label": "Main navigation"},
                    ),
                    html.Div(
                        [
                            badge("Compliance · v0.2"),
                            html.P("A foundation for evidence-based document review."),
                        ],
                        className="sidebar-footer",
                    ),
                ],
                className="sidebar",
            ),
            html.Div(
                [
                    html.Header(
                        [
                            html.Span("Document review / Compliance", className="breadcrumb"),
                            html.Div(
                                [
                                    html.Div(id="api-status", children=badge("Connecting…")),
                                    html.Button(
                                        "Refresh",
                                        id="refresh-button",
                                        n_clicks=0,
                                        className="button secondary",
                                    ),
                                ],
                                className="header-actions",
                            ),
                        ],
                        className="topbar",
                    ),
                    html.Div(id="workflow-status", role="status", className="notice"),
                    dcc.Loading(
                        html.Main(
                            id="page-content",
                            children=empty_state(
                                "Loading workspace",
                                "Connecting to your compliance API…",
                            ),
                        ),
                        type="circle",
                        color="#23705a",
                    ),
                ],
                className="main-shell",
            ),
        ],
        className="app-shell",
    )

    @app.callback(
        Output("page-content", "children"),
        Output("api-status", "children"),
        Output("navigation", "children"),
        Output("workflow-status", "children"),
        Input("url", "pathname"),
        Input("refresh-button", "n_clicks"),
        Input("refresh-interval", "n_intervals"),
        Input("workflow-version", "data"),
        Input("last-check", "data"),
    )
    def update_page(pathname, clicks, interval, workflow_version, last_check):
        pathname = pathname or "/"
        page, status = render_page(pathname, client, last_check)
        status_message = "" if ctx.triggered_id == "url" else no_update
        return page, status, navigation(pathname), status_message

    def upload(contents, filename, role):
        if not contents:
            return no_update, no_update
        try:
            content = base64.b64decode(contents.split(",", 1)[1], validate=True)
            document = client.upload(filename, content, role)
            return f"{document['filename']} is indexed and ready.", str(uuid4())
        except (APIUnavailable, ValueError, IndexError, binascii.Error) as exc:
            return str(exc), no_update

    for role, component in [("policy", "policy-upload"), ("subject", "subject-upload")]:

        def register_upload(role, component):
            @app.callback(
                Output("workflow-status", "children", allow_duplicate=True),
                Output("workflow-version", "data", allow_duplicate=True),
                Input(component, "contents"),
                State(component, "filename"),
                prevent_initial_call=True,
                running=[
                    (Output(component, "disabled"), True, False),
                    (Output("refresh-interval", "disabled"), True, False),
                ],
            )
            def on_upload(contents, filename):
                return upload(contents, filename, role)

        register_upload(role, component)

    @app.callback(
        Output("last-check", "data"),
        Output("workflow-status", "children", allow_duplicate=True),
        Input("run-check", "n_clicks"),
        State("subject-selection", "value"),
        State("policy-selection", "value"),
        State("specifications", "value"),
        prevent_initial_call=True,
        running=[
            (Output("run-check", "disabled"), True, False),
            (Output("refresh-interval", "disabled"), True, False),
        ],
    )
    def on_check(clicks, document_id, policy_ids, specifications):
        if not clicks:
            return no_update, no_update
        if not document_id or not policy_ids:
            return no_update, "Select a ready supplier document and at least one policy."
        try:
            result = client.create_check(
                document_id,
                policy_ids,
                [text.strip() for text in (specifications or "").splitlines() if text.strip()],
            )
            return result, "Compliance review completed."
        except APIUnavailable as exc:
            return no_update, str(exc)

    return app


app = create_app()
server = app.server

if __name__ == "__main__":
    settings = UISettings()
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
