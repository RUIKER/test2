"""
自动下载 NGAFID 基准子集数据。
无需额外下载依赖（仅使用 Python 标准库）。
"""
import json
import re
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

ALT_DATASET_SOURCES = [
    "https://doi.org/10.5281/zenodo.6624956",
    "https://www.kaggle.com/datasets/hooong/aviation-maintenance-dataset-from-the-ngafid",
]


def _extract_filename(content_disposition: str, default_name: str) -> str:
    filename_star = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
    if filename_star:
        return urllib.parse.unquote(filename_star.group(1)).strip()

    filename = re.search(r'filename="?([^";]+)"?', content_disposition, re.IGNORECASE)
    if filename:
        return filename.group(1).strip()

    return default_name


def _stream_to_file(response, output_path: Path):
    with open(output_path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _download_google_drive_file(file_id: str, download_dir: Path) -> Path:
    base_url = "https://drive.google.com/uc?export=download"
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    first_url = f"{base_url}&id={urllib.parse.quote(file_id)}"
    with opener.open(first_url, timeout=120) as response:
        content_disposition = response.headers.get("Content-Disposition", "")

        # 小文件可能首次请求就直接返回附件内容
        if "attachment" in content_disposition.lower():
            filename = _extract_filename(content_disposition, f"{file_id}.bin")
            output_path = download_dir / filename
            _stream_to_file(response, output_path)
            return output_path

        html = response.read().decode("utf-8", errors="ignore")

    # 大文件通常需要 confirm token
    confirm_match = re.search(r'confirm=([0-9A-Za-z_]+)', html)
    if not confirm_match:
        raise RuntimeError(
            f"无法获取 Google Drive 确认令牌，file_id={file_id}。可能是链接失效或网络受限。"
        )

    confirm_token = confirm_match.group(1)
    second_url = f"{base_url}&id={urllib.parse.quote(file_id)}&confirm={confirm_token}"
    with opener.open(second_url, timeout=120) as response:
        content_disposition = response.headers.get("Content-Disposition", "")
        filename = _extract_filename(content_disposition, f"{file_id}.bin")
        output_path = download_dir / filename
        _stream_to_file(response, output_path)
        return output_path


def _extract_drive_ids_from_text(text: str) -> list[str]:
    patterns = [
        r"https?://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    ]

    ids: list[str] = []
    for pattern in patterns:
        for file_id in re.findall(pattern, text):
            if file_id not in ids:
                ids.append(file_id)
    return ids


def _extract_file_ids_from_notebook(notebook_path: Path) -> list[str]:
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_content = json.load(f)

    ids: list[str] = []
    for cell in nb_content.get("cells", []):
        source = "".join(cell.get("source", []))
        for file_id in _extract_drive_ids_from_text(source):
            if file_id not in ids:
                ids.append(file_id)
    return ids


def _extract_file_ids_from_dataset_py(dataset_py_path: Path) -> list[str]:
    if not dataset_py_path.exists():
        return []

    text = dataset_py_path.read_text(encoding="utf-8")
    return _extract_drive_ids_from_text(text)


def _extract_named_file_ids_from_dataset_py(dataset_py_path: Path) -> dict[str, str]:
    if not dataset_py_path.exists():
        return {}

    text = dataset_py_path.read_text(encoding="utf-8")
    entries = re.findall(
        r'"([a-zA-Z0-9_]+)"\s*:\s*"https?://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)"',
        text,
    )
    return {name: file_id for name, file_id in entries}


def _has_local_subset_data(download_dir: Path) -> bool:
    return any(path.is_file() for path in download_dir.rglob("*"))

def extract_and_download_subset():
   # 1. 脚本当前在 test2/src 目录下，.parent 退回到 test2 根目录
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    # 2. 准确进入 data 目录寻找官方示例 Notebook（文件和 ngafiddataset 文件夹是平级的）
    repo_dir = project_root / "data"
    notebook_path = repo_dir / "NGAFID_DATASET_TF_EXAMPLE.ipynb"
    
    if not notebook_path.exists():
        raise FileNotFoundError(f"未找到 Notebook 文件: {notebook_path}")

    # 新建用于存放下载数据的文件夹，将其放在 data/subset_data 目录下比较整洁
    download_dir = project_root / "data" / "subset_data"
    download_dir.mkdir(parents=True, exist_ok=True)

    if _has_local_subset_data(download_dir):
        print(f"检测到本地数据已存在，跳过下载: {download_dir}")
        return

    print("正在解析下载链接...")

    # 3. 优先从 Notebook 提取 Google Drive 文件 ID
    file_ids = _extract_file_ids_from_notebook(notebook_path)

    # Notebook 可能不包含下载链接，回退到官方管理器源码中的 ngafid_urls
    if not file_ids:
        dataset_py_path = project_root / "data" / "ngafiddataset" / "dataset" / "dataset.py"
        named_ids = _extract_named_file_ids_from_dataset_py(dataset_py_path)
        if "2days" in named_ids:
            file_ids = [named_ids["2days"]]
        elif named_ids:
            file_ids = list(named_ids.values())
        else:
            file_ids = _extract_file_ids_from_dataset_py(dataset_py_path)

    if not file_ids:
        print("未找到可用下载链接，请检查 Notebook 或 data/ngafiddataset/dataset/dataset.py。")
        return

    # 4. 自动下载
    print(f"找到 {len(file_ids)} 个数据切片，准备下载到: {download_dir}")
    failed_ids = []
    for file_id in file_ids:
        print(f"正在下载 file_id={file_id} ...")
        try:
            saved_path = _download_google_drive_file(file_id, download_dir)
            print(f"已保存: {saved_path.name}")
        except Exception as exc:
            failed_ids.append(file_id)
            print(f"下载失败 file_id={file_id}: {exc}")

    if failed_ids:
        print(f"完成但有失败项: {failed_ids}")
        print("可尝试从以下官方来源手动获取数据后放入 data/subset_data:")
        for src in ALT_DATASET_SOURCES:
            print(f"- {src}")
    else:
        print("基准子集自动下载完成！")

if __name__ == "__main__":
    extract_and_download_subset()