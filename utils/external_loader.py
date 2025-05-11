import re
from pathlib import Path
from typing import List, Union
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

# Regex to find URLs in text
URL_PATTERN = re.compile(r"https?://[^\s)]+")


def extract_urls(text: str) -> List[str]:
    """
    Return a list of unique URLs found in the given text.
    """
    return list(set(URL_PATTERN.findall(text)))


def fetch_url_content(url: str) -> Union[str, Path]:
    """
    Fetch the URL and return its content.

    - If HTML, strip tags and return raw text.
    - If PDF (by Content-Type or URL suffix), download to .external_pdf_cache/ and return the Path.
    - Otherwise, return raw text.
    """
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")

    # HTML
    if "text/html" in content_type:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    # PDF
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        cache_dir = Path(".external_pdf_cache")
        cache_dir.mkdir(exist_ok=True)
        filename = cache_dir / Path(url).name
        filename.write_bytes(resp.content)
        return filename

    # Fallback to raw text
    return resp.text


def load_external_documents_from_pdfs(pdf_docs: List[Document]) -> List[Document]:
    """
    Scan a list of LangChain Documents for URLs, fetch each URL,
    and return new Document objects:
      - For HTML/text URLs, return Document(page_content=text, metadata={'source':url})
      - For PDF URLs, return Document(page_content='', metadata={'source':pdf_path})

    The ingest script can detect PDF paths in metadata['source'] and re-load them separately.
    """
    external_docs: List[Document] = []
    for doc in pdf_docs:
        urls = extract_urls(doc.page_content)
        for url in urls:
            try:
                content = fetch_url_content(url)
                if isinstance(content, Path):
                    # PDF downloaded to disk; placeholder Document
                    external_docs.append(
                        Document(page_content="", metadata={"source": str(content)})
                    )
                else:
                    # Plain text content
                    external_docs.append(
                        Document(page_content=content, metadata={"source": url})
                    )
            except Exception:
                # Skip URLs that fail to fetch
                continue
    return external_docs
