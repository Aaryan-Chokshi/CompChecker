from pathlib import Path

from dash import Dash, Input, Output, dcc, html

from ui.api.client import APIClient, APIUnavailable
from ui.components.common import badge, empty_state, unavailable_state
from ui.components.navigation import NAV_ITEMS, navigation
from ui.config import UISettings
from ui.features.compliance.layout import check_page, history_page
from ui.features.documents.layout import policy_page
from ui.features.overview import overview_page


def render_page(pathname: str, client: APIClient):
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
            page = check_page(client.documents(role="subject"), client.documents(role="policy"))
        else:
            page = history_page(client.checks())
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
    )
    app.layout = html.Div(
        [
            dcc.Location(id="url", refresh=False),
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
                            badge("Scaffold · v0.1"),
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
        Input("url", "pathname"),
        Input("refresh-button", "n_clicks"),
        Input("refresh-interval", "n_intervals"),
    )
    def update_page(pathname, clicks, interval):
        pathname = pathname or "/"
        page, status = render_page(pathname, client)
        return page, status, navigation(pathname)

    return app


app = create_app()
server = app.server

if __name__ == "__main__":
    settings = UISettings()
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
