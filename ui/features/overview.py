from dash import dcc, html

from ui.components.common import badge, page_header, workflow_notice


def overview_page(documents: list[dict], checks: list[dict], dependencies: dict):
    policies = sum(row.get("role") == "policy" for row in documents)
    subjects = sum(row.get("role") == "subject" for row in documents)
    stats = [
        ("Policies shown", policies),
        ("Documents shown", subjects),
        ("Recent checks", len(checks)),
    ]
    provider_rows = []
    for key, name in (
        ("database", "Application storage"),
        ("embeddings", "Local embeddings"),
        ("vector_store", "Policy vector index"),
        ("llm", "Compliance reasoning"),
    ):
        state = dependencies[key]
        if not state.get("configured"):
            label, tone = "Not configured", "warning"
        elif state.get("initialized"):
            label, tone = "Initialized", "success"
        else:
            label, tone = "Not initialized", "neutral"
        detail = state.get("model") or state.get("engine") or state.get("mode", "")
        provider_rows.append(
            html.Div(
                [
                    html.Div([html.Strong(name), html.P(detail, className="muted")]),
                    badge(label, tone),
                ],
                className="provider-row",
            )
        )
    return html.Div(
        [
            page_header(
                "Compliance workspace",
                "Bring your documents and policies into one review workflow.",
            ),
            workflow_notice(),
            html.Div(
                [
                    html.Div(
                        [
                            html.P(label, className="muted"),
                            html.Strong(str(value), className="stat-value"),
                        ],
                        className="card stat-card",
                    )
                    for label, value in stats
                ],
                className="stats-grid",
            ),
            html.P(
                "Counts reflect the latest 100 documents and checks loaded from the API.",
                className="caption",
            ),
            html.Div(
                [
                    html.Section(
                        [
                            html.P("YOUR WORKFLOW", className="eyebrow"),
                            html.H2("From policy to evidence"),
                            html.Ol(
                                [
                                    html.Li(
                                        [
                                            html.Strong("Build your policy library"),
                                            html.P(
                                                "Add the policies and regulations that guide your review."
                                            ),
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Strong("Choose a document"),
                                            html.P(
                                                "Select a document and the policies you want to check against."
                                            ),
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Strong("Review the findings"),
                                            html.P(
                                                "Inspect findings with supporting evidence, then decide on next steps."
                                            ),
                                        ]
                                    ),
                                ],
                                className="workflow-steps",
                            ),
                            dcc.Link(
                                "Open policy library →",
                                href="/policies",
                                className="text-link",
                            ),
                        ],
                        className="card",
                    ),
                    html.Section(
                        [
                            html.P("WORKSPACE STATUS", className="eyebrow"),
                            html.H2("Connected services"),
                            *provider_rows,
                            html.P(
                                "State only; external availability is not probed.",
                                className="caption",
                            ),
                        ],
                        className="card",
                    ),
                ],
                className="content-grid",
            ),
        ]
    )
