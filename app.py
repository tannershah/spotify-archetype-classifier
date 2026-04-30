"""
Vibe Check Dashboard
Plotly Dash multi-page app for the CIS 2450 final project.

Run with: python app.py   (opens http://127.0.0.1:8050)
"""

from dash import Dash, html, dcc, page_container, page_registry, Input, Output, callback

import dash

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Vibe Check — CIS 2450",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)

server = app.server  # for deployment

NAV_LINKS = [
    ("/",            "Home"),
    ("/pipeline",    "Pipeline"),
    ("/eda",         "EDA"),
    ("/performance", "Model"),
]

def navbar():
    return html.Nav(className="navbar", children=[
        html.Div(className="nav-inner", children=[
            html.A(className="brand", href="/", children=[
                html.Span("🎧", className="brand-emoji"),
                html.Span("Vibe Check", className="brand-name"),
            ]),
            html.Div(className="nav-links", children=[
                dcc.Link(label, href=path, className="nav-link",
                         id={"role": "nav", "path": path})
                for path, label in NAV_LINKS
            ]),
            html.Div(className="nav-meta", children=[
                html.Span("CIS 2450 · Spring 2026", className="nav-meta-text"),
            ]),
        ]),
    ])

def footer():
    return html.Footer(className="footer", children=[
        html.Div(className="footer-inner", children=[
            html.Span("Vibe Check"),
            html.Span("•", className="footer-sep"),
            html.Span("Kieran Chetty & Tanner Shah"),
            html.Span("•", className="footer-sep"),
            html.Span("CIS 2450 Spring 2026"),
        ]),
    ])

app.layout = html.Div(className="app-shell", children=[
    dcc.Location(id="url"),
    navbar(),
    html.Main(className="page", children=[page_container]),
    footer(),
])


# Highlight the active nav link based on current URL
@callback(
    Output({"role": "nav", "path": dash.ALL}, "className"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def _highlight_active(pathname):
    out = []
    for path, _ in NAV_LINKS:
        active = (pathname == path) or (pathname is None and path == "/")
        out.append("nav-link active" if active else "nav-link")
    return out


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
