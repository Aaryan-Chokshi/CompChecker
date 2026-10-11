from dash import dcc, html

from ui.components.common import (
    empty_state,
    page_header,
    records_table,
    workflow_notice,
)


def policy_page(policies: list[dict]):
    return html.Div(
        [
            page_header(
                "Policy Library",
                "The reference policies and regulations for your compliance checks.",
                html.Button("+ Add policy", disabled=True, className="button primary"),
            ),
            workflow_notice(),
            html.Section(
                [
                    html.H2("Add a policy document"),
                    dcc.Upload(
                        html.Div(
                            [
                                html.Strong("Policy uploads are coming next"),
                                html.P("Document parsing and indexing will be connected here."),
                            ]
                        ),
                        id="policy-upload",
                        disabled=True,
                        className="upload-zone disabled",
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
                        "Your policy library will appear here when ingestion is implemented.",
                    ),
                    html.P("Showing up to the latest 100 policies.", className="caption"),
                ],
                className="card",
            ),
        ]
    )
