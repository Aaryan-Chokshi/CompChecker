from dash import html


def badge(label: str, tone: str = "neutral"):
    return html.Span(label, className=f"badge {tone}")


def page_header(title: str, description: str, action=None):
    return html.Div(
        [html.Div([html.H1(title), html.P(description, className="muted")]), action],
        className="page-header",
    )


def empty_state(title: str, description: str):
    return html.Div(
        [html.Div("◇", className="empty-icon"), html.H3(title), html.P(description)],
        className="empty-state",
    )


def workflow_notice():
    return html.Div(
        [
            badge("Coming next", "warning"),
            html.Span(" Document ingestion and compliance evaluation are not implemented yet."),
        ],
        className="notice",
    )


def records_table(records: list[dict], columns: list[tuple[str, str]]):
    return html.Div(
        html.Table(
            [
                html.Thead(html.Tr([html.Th(label, scope="col") for _, label in columns])),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(
                                    badge(str(record.get(key, "—")).replace("_", " "))
                                    if key == "status"
                                    else str(record.get(key) or "—")
                                )
                                for key, _ in columns
                            ]
                        )
                        for record in records
                    ]
                ),
            ],
            className="records-table",
        ),
        className="table-scroll",
    )


def unavailable_state(message: str):
    return html.Div(
        [
            html.H2("Unable to load this view"),
            html.P(message),
            html.P("Use Refresh to try again."),
        ],
        className="notice error-state",
        role="alert",
    )
