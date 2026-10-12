from dash import dcc, html

from ui.components.common import (
    empty_state,
    page_header,
    records_table,
)


def check_page(subjects: list[dict], policies: list[dict], last_check=None, client=None):
    return html.Div(
        [
            page_header(
                "Compliance Check",
                "Review a document against the policies that apply to it.",
            ),
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
                                    if r.get("status") == "ready"
                                ],
                                placeholder="Select a document",
                                persistence=True,
                                persistence_type="session",
                            ),
                            dcc.Upload(
                                html.Div(
                                    "Drop a supplier specification document or click to select"
                                ),
                                id="subject-upload",
                                accept=".pdf,.docx,.txt,.md,.csv,.tsv",
                                className="upload-zone compact",
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
                                    if r.get("status") == "ready"
                                ],
                                multi=True,
                                placeholder="Select policies from your library",
                                persistence=True,
                                persistence_type="session",
                            ),
                            html.Label(
                                "Additional specifications (optional)", htmlFor="specifications"
                            ),
                            dcc.Textarea(
                                id="specifications",
                                placeholder="One specification per line",
                                persistence=True,
                                persistence_type="session",
                                style={"width": "100%", "minHeight": "90px"},
                            ),
                            html.Button(
                                "Run compliance check",
                                id="run-check",
                                className="button primary wide",
                            ),
                            html.P(
                                "Processing uses local embeddings and Ollama for chunking and review. Large reviews may take several minutes.",
                                className="caption",
                            ),
                        ],
                        className="card",
                    ),
                    html.Section(
                        [
                            html.H2("Findings"),
                            result_view(last_check, client)
                            if last_check and client
                            else empty_state(
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


def history_page(checks: list[dict], client=None):
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
                        "Your compliance reviews will appear here after you run a check.",
                    ),
                    html.P("Showing up to the latest 100 checks.", className="caption"),
                    *[
                        html.Details(
                            [html.Summary(f"Review {check['id']}"), result_view(check, client)]
                        )
                        for check in checks
                        if check.get("result") and client
                    ],
                ],
                className="card",
            ),
        ]
    )


def result_view(check, client):
    result = check.get("result") or {}
    rows = result.get("comparison_matrix", [])

    def evidence(items):
        if not items:
            return html.Span("No citations supplied by the reviewer", className="caption")
        return html.Ul(
            [
                html.Li(
                    f"{item.get('filename', '')}: {item.get('excerpt', '')} "
                    + " ".join(
                        f"{key} {item.get('provenance', {}).get(key)}"
                        for key in ("page", "table", "row")
                        if item.get("provenance", {}).get(key) is not None
                    ),
                    title=f"Source chunk: {item.get('chunk_id', '')}",
                )
                for item in items
            ]
        )

    def evidence_cell(row, source):
        candidates = row.get(f"retrieved_{source}_chunks", [])
        return html.Div(
            [
                evidence(row.get(f"{source}_evidence", [])),
                *(
                    [
                        html.Details(
                            [
                                html.Summary(
                                    f"Source context supplied to reviewer ({len(candidates)})"
                                ),
                                html.P(
                                    "These are source candidates, not confirmed supporting citations.",
                                    className="caption",
                                ),
                                evidence(candidates),
                            ]
                        )
                    ]
                    if candidates
                    else []
                ),
            ]
        )

    return html.Div(
        [
            html.H3(str(result.get("verdict", "needs_review")).replace("_", " ").title()),
            html.P(
                f"Review {check['id']} · Supplier document {check.get('document_id', '')}",
                className="caption",
            ),
            html.P(result.get("summary", "")),
            html.P(
                "Extent is estimated satisfaction from 0–100%. Confidence is the LLM's self-assessed decision confidence; it is not a calibrated probability.",
                className="caption",
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr(
                                [
                                    html.Th(label)
                                    for label in [
                                        "Specification or requirement",
                                        "Regulatory obligation",
                                        "Satisfaction",
                                        "Extent",
                                        "Confidence",
                                        "Reason",
                                        "Policy evidence",
                                        "Supplier evidence",
                                    ]
                                ]
                            )
                        ),
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(row["specification"]),
                                        html.Td(row.get("requirement", "")),
                                        html.Td(row["satisfaction"].replace("_", " ")),
                                        html.Td(
                                            f"{row['extent_percent']:g}%"
                                            if row.get("extent_percent") is not None
                                            else "Unknown / N/A"
                                        ),
                                        html.Td(f"{row['confidence']:.0%}"),
                                        html.Td(row["explanation"]),
                                        html.Td(evidence_cell(row, "policy")),
                                        html.Td(evidence_cell(row, "supplier")),
                                    ]
                                )
                                for row in rows
                            ]
                        ),
                    ],
                    className="records-table",
                ),
                className="table-scroll",
            ),
            html.H3("Supplier clarification sheet"),
            html.Div(
                [
                    html.A(
                        f"Download {format.upper()}",
                        href=client.clarification_url(check["id"], format),
                        className="button secondary",
                        target="_blank",
                        style={"marginRight": "8px"},
                    )
                    for format in ["md", "docx", "pdf"]
                ]
            ),
            dcc.Markdown(result.get("clarification_markdown", "")),
        ]
    )
