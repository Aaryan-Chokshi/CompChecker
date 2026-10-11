from dash import dcc, html

from ui.components.common import (
    empty_state,
    page_header,
    records_table,
)


def policy_page(policies: list[dict]):
    return html.Div(
        [
            page_header(
                "Policy Library",
                "The reference policies and regulations for your compliance checks.",
            ),
            html.Section(
                [
                    html.H2("Add a policy document"),
                    dcc.Upload(
                        html.Div(
                            [
                                html.Strong("Drop a policy document or click to select"),
                                html.P(
                                    "PDF, DOCX, TXT, Markdown, CSV or TSV. First use may download the local embedding model."
                                ),
                            ]
                        ),
                        id="policy-upload",
                        accept=".pdf,.docx,.txt,.md,.csv,.tsv",
                        className="upload-zone",
                    ),
                ],
                className="card",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Your policies"),
                            html.Span(f"{len(policies)} shown", className="muted"),
                        ],
                        className="section-header",
                    ),
                    records_table(
                        policies,
                        [
                            ("filename", "Policy document"),
                            ("status", "Status"),
                            ("created_at", "Added (UTC)"),
                        ],
                    )
                    if policies
                    else empty_state(
                        "No policies yet",
                        "Upload a regulation or policy document to build your library.",
                    ),
                    html.P("Showing up to the latest 100 policies.", className="caption"),
                ],
                className="card",
            ),
        ]
    )
