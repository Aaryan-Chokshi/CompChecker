from dash import dcc, html

NAV_ITEMS = (
    ("/", "Overview", "01"),
    ("/policies", "Policy Library", "02"),
    ("/check", "Compliance Check", "03"),
    ("/history", "History", "04"),
)


def navigation(pathname: str):
    return [
        dcc.Link(
            [html.Span(number, className="nav-number"), html.Span(label)],
            href=path,
            className="nav-link active" if pathname == path else "nav-link",
        )
        for path, label, number in NAV_ITEMS
    ]
