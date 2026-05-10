from __future__ import annotations

import argparse
import json
import pathlib

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn


def load_json(path: pathlib.Path):
    if not path.exists():
        return {"error": f"missing {path}"}
    with path.open() as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="runs/gpu_report.json")
    parser.add_argument("--token-meta", default="data/fineweb_gpt2_10m_meta.json")
    parser.add_argument("--loss-png", default="runs/gpu_report_loss.png")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    report_path = pathlib.Path(args.report)
    token_meta_path = pathlib.Path(args.token_meta)
    loss_png_path = pathlib.Path(args.loss_png)

    app = FastAPI(title="efficient-ngpt RunPod report")

    @app.get("/health")
    def health():
        report = load_json(report_path)
        return {
            "ok": bool(report.get("comparison", {}).get("all_gates", False)),
            "report": str(report_path),
            "token_meta": str(token_meta_path),
        }

    @app.get("/report")
    def report():
        return JSONResponse(load_json(report_path))

    @app.get("/tokens")
    def tokens():
        return JSONResponse(load_json(token_meta_path))

    @app.get("/loss.png")
    def loss_png():
        return FileResponse(loss_png_path)

    @app.get("/")
    def index():
        report = load_json(report_path)
        meta = load_json(token_meta_path)
        comparison = report.get("comparison", {})
        html = f"""
        <html>
          <head>
            <title>efficient-ngpt RunPod report</title>
            <style>
              body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 980px; margin: 32px auto; line-height: 1.45; }}
              code, pre {{ background: #f4f4f5; padding: 2px 4px; border-radius: 4px; }}
              pre {{ padding: 16px; overflow: auto; }}
              .ok {{ color: #047857; font-weight: 700; }}
              .bad {{ color: #b91c1c; font-weight: 700; }}
            </style>
          </head>
          <body>
            <h1>efficient-ngpt RunPod report</h1>
            <p>Status: <span class="{'ok' if comparison.get('all_gates') else 'bad'}">{comparison.get('all_gates')}</span></p>
            <h2>Comparison</h2>
            <pre>{json.dumps(comparison, indent=2, sort_keys=True)}</pre>
            <h2>Tokenization</h2>
            <pre>{json.dumps(meta, indent=2, sort_keys=True)}</pre>
            <h2>Loss</h2>
            <img src="/loss.png" style="max-width: 100%; border: 1px solid #e4e4e7;" />
            <p>JSON endpoints: <a href="/health">/health</a>, <a href="/report">/report</a>, <a href="/tokens">/tokens</a>, <a href="/loss.png">/loss.png</a></p>
          </body>
        </html>
        """
        return HTMLResponse(html)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
