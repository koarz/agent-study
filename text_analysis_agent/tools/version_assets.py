"""按文件内容更新资源版本号，避免浏览器继续使用旧版前端脚本和样式。"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path


def update_assets(web: Path):
    index = web / "index.html"
    original = index.read_text(encoding="utf-8")
    updated = original
    for name in ("app.js", "styles.css"):
        version = hashlib.sha256((web / name).read_bytes()).hexdigest()[:12]
        path = "/assets/" + name
        updated = re.sub(re.escape(path) + r'(?:\?v=[^"\s<>]*)?', path + "?v=" + version, updated)
    if updated != original:
        # 仅内容变化时替换页面，避免每次启动都改变修改时间。
        with tempfile.NamedTemporaryFile(dir=web, prefix=".index-", suffix=".tmp", delete=False) as writer:
            temporary = Path(writer.name)
            try:
                writer.write(updated.encode("utf-8"))
                writer.flush()
                os.chmod(temporary, index.stat().st_mode & 0o777)
                temporary.replace(index)
            finally:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    update_assets(Path(__file__).resolve().parents[1] / "web")
