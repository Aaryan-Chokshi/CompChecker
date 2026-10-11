from dash import dcc, html

from ui.components.common import (
    empty_state,
    page_header,
    records_table,
    workflow_notice,
)


def check_page(subjects: list[dict], policies: list[dict]):
    return html.Div(
        [
            page_header(
                "Compliance Check",
                "Review a document against the policies that apply to it.",
            ),
            workflow_notice(),
            html.Div(
                [
                    html.Section(
                        [
                            html.H2("Prepare your review"),
                            html.Label("Document to review", htmlFor="subject-selection"),
                            dcc.Dropdown(
                                id="subject-selection",
                                options=[
                                    {
                                        "label": r.get("filename", "Document"),
                                        "value": r.get("id"),
                                    }
                                    for r in subjects
                                ],
                                placeholder="Select a document",
                                disabled=True,
                            ),
                            dcc.Upload(
                                html.Div("Document uploads are coming next"),
                                id="subject-upload",
                                disabled=True,
                                className="upload-zone disabled compact",
                            ),
                            html.Label("Policies to apply", htmlFor="policy-selection"),
                            dcc.Dropdown(
                                id="policy-selection",
                                options=[
                                    {
                                        "label": r.get("filename", "Policy"),
                                        "value": r.get("id"),
                                    }
                                    for r in policies
                                ],
                                multi=True,
                                placeholder="Select policies from your library",
                                disabled=True,
                            ),
                            html.Button(
                                "Run compliance check",
                                id="run-check",
                                disabled=True,
                                className="button primary wide",
                            ),
                            html.P(
                                "Review controls will be enabled once ingestion and evaluation are connected.",
                                className="caption",
                            ),
                        ],
                        className="card",
                    ),
                    html.Section(
                        [
                            html.H2("Findings"),
                            empty_state(
                                "No check has been run",
                                "Findings and supporting policy evidence will appear here after a completed check.",
                            ),
                        ],
                        className="card",
                    ),
                ],
                className="content-grid",
            ),
        ]
    )


def history_page(checks: list[dict]):
    return html.Div(
        [
            page_header("History", "A record of compliance checks and their results."),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Recent checks"),
                            html.Span(f"{len(checks)} shown", className="muted"),
                        ],
                        className="section-header",
                    ),
                    records_table(
                        checks,
                        [
                            ("id", "Check ID"),
                            ("document_id", "Document ID"),
                            ("status", "Status"),
                            ("created_at", "Started (UTC)"),
                        ],
                    )
                    if checks
                    else empty_state(
                        "No checks yet",
                        "Completed and in-progress checks will be listed here when evaluation is implemented.",
                    ),
                    html.P("Showing up to the latest 100 checks.", className="caption"),
                ],
                className="card",
            ),
        ]
    )
