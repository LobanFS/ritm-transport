"""Собирает переносимую документацию PyDoc и OpenAPI из текущего кода."""
from html import escape, unescape
import importlib
import json
from pathlib import Path
import pydoc
import re
import sys
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPOSITORY = "https://github.com/LobanFS/ritm-transport"
GROUPS = {
    "Общие контракты": ["common", "common.contracts", "common.state", "common.transfer"],
    "Backend и входные данные": [
        "backend", "backend.app", "backend.engine", "backend.ndtp",
        "backend.arrivals", "backend.gps_arrivals", "backend.replay",
        "backend.diagnostics", "backend.segment_observations", "backend.risk_summary",
        "backend.generator_bridge", "backend.learning_store", "backend.transfer_advisor",
    ],
    "ML-сервис": [
        "ml_service", "ml_service.app", "ml_service.learned", "ml_service.hybrid",
        "ml_service.artifacts", "ml_service.probability", "ml_service.explainability",
        "ml_service.frozen_plan", "ml_service.frozen_plan.plan",
    ],
    "Подготовка новой версии модели": [
        "training", "training.export", "training.production_data",
        "training.refit_hgbr", "training.scheduler",
    ],
    "Синтетические сценарии": ["generator", "generator.scenarios", "generator.app"],
}
MODULES = {name for names in GROUPS.values() for name in names}
LOCAL_MODULE_PAGES = {f"{name}.html" for name in MODULES} | {"index.html"}


def remove_unpublished_links(html, page_name=""):
    """Оставить текст внешних типов, на которые PyDoc сам ставит локальные ссылки.

    Наследуемые методы и дескрипторы получают ссылки в обход classlink.
    Их страницы зависимостей в эту документацию не входят.
    """
    anchors = {unescape(match[2]) for match in re.finditer(
        r'''\b(?:id|name)\s*=\s*(["'])(.*?)\1''', html, re.IGNORECASE | re.DOTALL)}

    def clean(match):
        href = re.search(r'''\bhref\s*=\s*(["'])(.*?)\1''', match[1], re.IGNORECASE | re.DOTALL)
        if href is not None:
            target = urlsplit(unescape(href[2]))
            if (not target.scheme and not target.netloc
                    and target.path.endswith(".html")
                    and target.path not in LOCAL_MODULE_PAGES):
                return match[2]
            if (not target.scheme and not target.netloc
                    and target.path in ("", page_name)
                    and target.fragment and target.fragment not in anchors):
                return match[2]
        return match[0]

    return re.sub(r"<a\b([^>]*)>(.*?)</a\s*>", clean, html, flags=re.IGNORECASE | re.DOTALL)


class PortableHTMLDoc(pydoc.HTMLDoc):
    """Ссылки ведут в репозиторий, а не на компьютер автора сборки."""

    def filelink(self, url, path):
        try:
            relative = Path(path).resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return escape(Path(path).name)
        return f'<a href="{REPOSITORY}/blob/main/{quote(relative)}">{escape(relative)}</a>'

    def modulelink(self, module):
        return super().modulelink(module) if module.__name__ in MODULES else escape(module.__name__)

    def classlink(self, cls, modname):
        return (super().classlink(cls, modname) if cls.__module__ in MODULES
                else escape(pydoc.classname(cls, modname)))

    def modpkglink(self, info):
        name, parent, ispackage, shadowed = info
        qualified = f"{parent}.{name}" if parent else name
        return super().modpkglink(info) if qualified in MODULES else escape(name)


def index_html(prefix=""):
    sections = "".join(
        f"<section><h2>{escape(title)}</h2><ul>" + "".join(
            f'<li><a href="{prefix}{name}.html">{name}</a></li>' for name in names
        ) + "</ul></section>" for title, names in GROUPS.items()
    )
    return """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ритм — документация</title><style>
body{max-width:1080px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui,sans-serif;color:#243942;background:#f7faf9}
a{color:#176c62}h1{margin-bottom:8px}h2{font-size:20px}.modules{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
section{padding:18px 24px;border:1px solid #d8e3df;border-radius:12px;background:white}ul{padding-left:20px}
</style></head><body><h1>Ритм — документация</h1>
<p>PyDoc и схемы OpenAPI собраны из исходников этой версии. Сервисы при генерации не запускаются.</p>
""" + f"""<p><a href="{REPOSITORY}/blob/main/docs/JURY.md">Инструкция для жюри</a> ·
<a href="{REPOSITORY}/blob/main/docs/OPERATIONS.md">Входные данные и эксплуатация</a></p>
<h2>API</h2><p>После локального запуска системы:
<a href="http://127.0.0.1:8000/docs">Backend Swagger</a> · <a href="http://127.0.0.1:8001/docs">ML Swagger</a>.</p>
<p>Схемы для просмотра и импорта: <a href="{prefix}backend.openapi.json">Backend OpenAPI</a> ·
<a href="{prefix}ml.openapi.json">ML OpenAPI</a> · <a href="{prefix}generator.openapi.json">Generator OpenAPI</a>.</p>
<h2>Модули</h2><div class="modules">{sections}</div></body></html>\n"""


def main():
    out = ROOT / "docs" / "reference"
    out.mkdir(parents=True, exist_ok=True)
    from backend.app import app as backend
    from ml_service.app import app as ml
    from generator.app import app as generator

    for name, app in [("backend", backend), ("ml", ml), ("generator", generator)]:
        (out / f"{name}.openapi.json").write_text(
            json.dumps(app.openapi(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    doc = PortableHTMLDoc()
    for name in sorted(MODULES):
        html = doc.page(name, doc.document(importlib.import_module(name), name))
        html = remove_unpublished_links(html, f"{name}.html")
        # PyDoc также печатает значения констант Path, например ROOT.
        html = html.replace(str(ROOT), ".")
        (out / f"{name}.html").write_text("\n".join(line.rstrip() for line in html.splitlines()) + "\n", encoding="utf-8")

    (out / "index.html").write_text(index_html(), encoding="utf-8")
    (out.parent / "index.html").write_text(index_html("reference/"), encoding="utf-8")
    (out.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(out / "index.html")


if __name__ == "__main__":
    main()
