#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTChar, LTTextContainer, LTTextLine


FOLDERS = [
    "ファクター研究",
    "ファクター値",
    "その他レポート",
    "書類等",
    "哲学",
    "不動産投資",
    "会計",
    "投資",
    "経済",
    "統計",
]

PDF_EXTENSIONS = {".pdf"}
RENAME_ALLOWED_FOLDERS = {"ファクター研究", "統計", "投資", "経済", "会計", "不動産投資", "その他レポート"}
MOVE_ALLOWED_TARGETS = {
    "ファクター研究": {"統計", "投資"},
    "その他レポート": {"ファクター研究"},
}
INVALID_FILENAME_CHARS = r'<>:"/\|?*'
GENERIC_STEM_PATTERNS = [
    re.compile(r"^(ssrn[-_ ]?\d+|gd\d+|cv[_-].+|d\d+|t\d+[-_]\d+|[0-9_]+)$", re.I),
    re.compile(r"^[A-Z]{2,}\d{3,}.*$"),
]
TITLE_SKIP_PATTERNS = [
    re.compile(pattern, re.I)
    for pattern in [
        r"^journal of ",
        r"^contents lists available",
        r"^journal homepage",
        r"^doi[: ]",
        r"^received ",
        r"^accepted ",
        r"^available online",
        r"^abstract$",
        r"^a b s t r a c t$",
        r"^keywords?$",
        r"^要約$",
        r"^現代ファイナンス",
        r"^no\.?\s*\d+",
        r"^doi[:：]",
        r"^\(\d{4}\)$",
        r"^volume \d+",
        r"^number \d+",
        r"^issue date",
        r"^january$",
        r"^february$",
        r"^march$",
        r"^april$",
        r"^may$",
        r"^june$",
        r"^july$",
        r"^august$",
        r"^september$",
        r"^october$",
        r"^november$",
        r"^december$",
        r"^博士学位論文$",
        r"^修士学位論文$",
        r"^題\s*名$",
        r"^title$",
        r"^summary$",
        r"^キーワード",
        r"^keywords?[:：]?",
        r"^要$",
        r"^約$",
        r"^氏名$",
        r"^学籍番号$",
        r"^指導教員",
        r"^submitted ",
        r"^http[s]?://",
        r"^www\.",
    ]
]
CATEGORY_KEYWORDS = {
    "統計": {
        "回帰": 5,
        "重回帰": 6,
        "多重回帰": 6,
        "主成分": 6,
        "principal component": 6,
        "statistical": 4,
        "statistics": 4,
        "推定量": 3,
        "likelihood": 4,
        "尤度": 4,
        "filtering": 4,
        "prediction problems": 4,
        "garch": 5,
        "arch": 5,
        "heteroskedasticity": 5,
        "kalman": 4,
        "closest fit": 5,
        "analysis of a complex of statistical variables": 6,
        "空間回帰": 5,
        "解析": 3,
        "分析": 2,
        "newey-west": 4,
        "markov-switching": 2,
    },
    "投資": {
        "ポートフォリオ": 5,
        "portfolio": 5,
        "asset allocation": 4,
        "investment strategy": 4,
        "投資戦略": 4,
        "investor": 2,
        "stock market": 2,
        "stock markets": 2,
        "stock prices": 2,
        "資産価格": 3,
        "asset pricing": 3,
        "mean reversion": 3,
        "portfolio factor exposures": 5,
    },
    "経済": {
        "景気循環": 5,
        "business cycle": 5,
        "金融政策": 5,
        "monetary policy": 5,
        "バブル": 5,
        "bubble": 5,
        "inflation": 4,
        "失われた": 3,
        "通貨": 2,
        "bull": 2,
        "bear": 2,
        "景気": 3,
    },
    "ファクター研究": {
        "ファクター": 6,
        "factor": 6,
        "momentum": 4,
        "anomalies": 3,
        "residual": 3,
        "stock returns": 3,
        "comovement": 3,
        "value anomaly": 3,
        "size anomaly": 3,
        "liquidity factor": 4,
    },
}


@dataclass
class FileInfo:
    path: str
    folder: str
    extension: str
    size: int
    mtime: float
    sha256: str
    current_stem: str
    extracted_title: Optional[str] = None
    title_method: Optional[str] = None
    title_confidence: float = 0.0
    normalized_title: Optional[str] = None
    text_fingerprint: Optional[str] = None
    category_scores: dict[str, int] = field(default_factory=dict)
    predicted_folder: Optional[str] = None
    predicted_score: int = 0
    move_reason: Optional[str] = None
    desired_stem: Optional[str] = None


def normalize_spaces(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\u3000", " ")
    return re.sub(r"\s+", " ", value).strip()


def normalize_key(value: Optional[str]) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^0-9a-zぁ-んァ-ヶ一-龯]+", "", value)


def sanitize_filename(value: str) -> str:
    value = normalize_spaces(value)
    value = "".join("_" if char in INVALID_FILENAME_CHARS else char for char in value)
    value = value.rstrip("._ ")
    return value[:180].strip() or "untitled"


def truncate_title(value: str, threshold: int = 80) -> str:
    value = normalize_spaces(value)
    if len(value) <= threshold:
        return value
    for index, char in enumerate(value):
        if char in "。．.!?！？:：;；,，、" and index >= 10:
            trimmed = value[: index + 1].strip()
            if trimmed:
                return trimmed
    return value[:threshold].strip()


def is_generic_stem(stem: str) -> bool:
    stem = normalize_spaces(stem)
    ascii_like = re.fullmatch(r"[A-Za-z0-9._\-\s]+", stem or "")
    if len(stem) <= 6 and ascii_like:
        return True
    return any(pattern.match(stem) for pattern in GENERIC_STEM_PATTERNS)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_font_size(line: LTTextLine) -> float:
    sizes: list[float] = []
    for child in line:
        if isinstance(child, LTChar):
            sizes.append(round(child.size, 2))
    return max(sizes) if sizes else 0.0


def title_blacklisted(text: str) -> bool:
    text = normalize_spaces(text)
    return any(pattern.search(text) for pattern in TITLE_SKIP_PATTERNS)


def score_title_line(text: str, font_size: float = 0.0) -> float:
    text = normalize_spaces(text)
    if not text:
        return -999.0
    if len(text) < 4 or len(text) > 220:
        return -999.0
    if title_blacklisted(text):
        return -8.0
    score = 0.0
    alpha = sum(char.isalpha() for char in text)
    digits = sum(char.isdigit() for char in text)
    spaces = sum(char.isspace() for char in text)
    ratio = alpha / max(len(text) - spaces, 1)
    score += ratio * 3.0
    score += min(font_size / 4.0, 4.0)
    if re.search(r"[A-Za-zぁ-んァ-ヶ一-龯]", text):
        score += 1.0
    if not re.search(r"[.;,]$", text):
        score += 0.5
    if digits and digits / max(len(text), 1) > 0.25:
        score -= 2.0
    if text.isupper():
        score += 0.8
    if re.search(r"(university|department|school|faculty|received|accepted)", text, re.I):
        score -= 2.5
    if re.search(r"(abstract|要約|keyword|journal|contents)", text, re.I):
        score -= 3.0
    return score


def clean_title_candidate(value: str) -> str:
    value = normalize_spaces(value)
    value = re.sub(r"[\*$†‡]+$", "", value).strip()
    value = re.sub(r"\s+[•●]\s*$", "", value).strip()
    value = re.sub(r"^Title[-:]\s*", "", value, flags=re.I)
    value = re.sub(r"\s*Author\(s\).*?$", "", value, flags=re.I)
    value = re.sub(r"\s*Source[:：].*$", "", value, flags=re.I)
    value = re.sub(r"\s*http[s]?://\S+$", "", value, flags=re.I)
    value = re.sub(r"\s+メタデータ.*$", "", value)
    value = re.sub(r"\s+言語[:：_ ].*$", "", value)
    value = re.sub(r"\s+[A-Z][a-z]+,\s*[A-Z][a-z]+$", "", value)
    value = re.sub(r"\s+[一-龯ぁ-んァ-ヶ]{1,4},\s*[一-龯ぁ-んァ-ヶ]{1,4}$", "", value)
    value = re.sub(r"\s+[一-龯]{1,4}(?:\s+[一-龯]{1,4}){1,2}$", "", value)
    value = re.sub(r"(?<=[A-Za-zぁ-んァ-ヶ一-龯])\d$", "", value)
    return value


def text_skip_line(text: str) -> bool:
    text = normalize_spaces(text)
    if title_blacklisted(text):
        return True
    if text in {"●", "•", "-", "―"}:
        return True
    if re.search(r"^(?:19|20)\d{2}$", text):
        return True
    if re.search(r"^(?:19|20)\d{2}\s*年", text):
        return True
    if re.search(r"^\d+[〜~\-]\d+$", text):
        return True
    if re.search(r"^(volume|vol\.?|number|no\.?)\b", text, re.I):
        return True
    if re.search(r"(jpm\.pm-research\.com|science direct|elsevier)", text, re.I):
        return True
    return False


def looks_author_line(text: str) -> bool:
    text = normalize_spaces(text)
    if re.search(r"(university|department|school|faculty|大学|研究科|准教授|教授)", text, re.I):
        return True
    if re.search(r"(受付|受理|提出|submitted|received|accepted)", text, re.I):
        return True
    if re.search(r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,4}$", text):
        return True
    if text.count(",") >= 2 and len(re.findall(r"[A-Z][a-z]+", text)) >= 4:
        return True
    return False


def looks_name_like(text: str) -> bool:
    text = normalize_spaces(text)
    if re.fullmatch(r"[一-龯ぁ-んァ-ヶ]{2,8}", text):
        return True
    if re.fullmatch(r"[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}", text):
        return True
    if re.fullmatch(r"[A-Z]{2,}(?: [A-Z]{2,}){1,2}", text) and len(text) <= 30:
        return True
    return False


def is_continuation_line(text: str) -> bool:
    text = normalize_spaces(text).casefold()
    return text.startswith("and ") or text.startswith("or ")


def title_stop_line(text: str) -> bool:
    text = normalize_spaces(text)
    if looks_author_line(text):
        return True
    if re.search(r"^(a r t i c l e i n f o|abstract|a b s t r a c t|summary)$", text, re.I):
        return True
    if re.search(r"^(keywords?|キーワード)[:：]?", text, re.I):
        return True
    if re.search(r"^(頁|page)\b", text, re.I):
        return True
    if re.search(r"^(指導教員|学籍番号|氏名)", text):
        return True
    return False


def special_case_title_from_line(text: str) -> Optional[str]:
    text = normalize_spaces(text)
    match = re.search(r"\b(?:18|19|20)\d{2}\.\s+(.+?)\.\s+[A-Z][^.]{2,}", text)
    if match:
        return clean_title_candidate(match.group(1))
    return None


def candidate_quality(text: str) -> float:
    text = clean_title_candidate(text)
    if not text:
        return -999.0
    if re.search(r"(cid_|http[s]?://|www\.)", text, re.I):
        return -10.0
    score = score_title_line(text)
    if is_continuation_line(text):
        score -= 3.0
    if text_skip_line(text):
        score -= 8.0
    if looks_author_line(text):
        score -= 3.0
    if re.fullmatch(r"[A-Z ]{2,25}", text):
        score -= 1.5
    if len(text) < 8:
        score -= 2.0
    if len(text) > 160:
        score -= 2.0
    elif len(text) > 110:
        score -= 4.0
    elif len(text) > 85:
        score -= 2.0
    if re.search(r"[,:：―\-]", text):
        score += 0.3
    if re.search(r"(博士学位論文|修士学位論文|summary|issue date)", text, re.I):
        score -= 6.0
    if re.search(r"\b\d+(?:st|nd|rd|th) edition\b", text, re.I):
        score -= 3.0
    letters = [char for char in text if char.isalpha()]
    if letters:
        upper_ratio = sum(char.isupper() for char in letters) / len(letters)
        if upper_ratio > 0.75 and len(text.split()) <= 5:
            score -= 2.0
    return score


def title_looks_usable(title: str) -> bool:
    title = normalize_spaces(title)
    if len(normalize_key(title)) < 8:
        return False
    bad_patterns = [
        r"cid_",
        r"citation",
        r"principal purpose",
        r"terms of use",
        r"affiliation",
        r"email[:_ ]",
        r"ふりがな",
        r"ご担当者様",
        r"maintain personal history",
        r"研究成果報告書",
        r"学位論文題目",
        r"link terms",
        r"issue date",
        r"metadata",
    ]
    if any(re.search(pattern, title, re.I) for pattern in bad_patterns):
        return False
    return candidate_quality(title) >= 2.8


def extract_title_from_layout(path: Path) -> tuple[Optional[str], float]:
    lines: list[dict[str, float | str]] = []
    try:
        for page_layout in extract_pages(str(path), maxpages=1):
            for element in page_layout:
                if not isinstance(element, LTTextContainer):
                    continue
                for line in element:
                    if not isinstance(line, LTTextLine):
                        continue
                    text = clean_title_candidate(line.get_text())
                    if not text:
                        continue
                    font_size = line_font_size(line)
                    lines.append(
                        {
                            "text": text,
                            "font_size": font_size,
                            "y": round(getattr(line, "y1", 0.0), 2),
                            "x": round(getattr(line, "x0", 0.0), 2),
                        }
                    )
            break
    except Exception:
        return None, 0.0

    if not lines:
        return None, 0.0

    ranked = []
    for line in lines:
        score = score_title_line(str(line["text"]), float(line["font_size"]))
        ranked.append({**line, "score": score})

    ranked.sort(key=lambda item: (item["score"], item["font_size"], item["y"]), reverse=True)
    best = ranked[0]
    if best["score"] < 2.5:
        return None, 0.0

    neighbors = [best]
    for line in ranked[1:]:
        if line["score"] < 1.0:
            continue
        if abs(float(line["font_size"]) - float(best["font_size"])) > 1.8:
            continue
        if abs(float(line["x"]) - float(best["x"])) > 30:
            continue
        if abs(float(line["y"]) - float(best["y"])) > max(float(best["font_size"]) * 2.2, 28):
            continue
        neighbors.append(line)

    neighbors.sort(key=lambda item: float(item["y"]), reverse=True)
    candidate = " ".join(str(item["text"]) for item in neighbors[:3]).strip()
    candidate = clean_title_candidate(candidate)
    if candidate_quality(candidate) < 3.0:
        return None, 0.0
    confidence = min(0.8, 0.35 + float(best["score"]) / 14.0)
    return candidate or None, confidence


def extract_title_from_text(path: Path) -> tuple[Optional[str], float]:
    try:
        text = extract_text(str(path), maxpages=1) or ""
    except Exception:
        return None, 0.0

    raw_lines = [clean_title_candidate(line) for line in text.splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return None, 0.0

    for line in lines[:25]:
        special = special_case_title_from_line(line)
        if special and candidate_quality(special) >= 2.5:
            return special, 0.74

    limited_lines = lines[:12]
    candidates: list[tuple[float, str]] = []

    for index, line in enumerate(limited_lines):
        if not title_stop_line(line):
            continue
        pieces: list[str] = []
        back_index = index - 1
        while back_index >= 0 and len(pieces) < 3:
            previous = limited_lines[back_index]
            if text_skip_line(previous):
                if pieces:
                    break
                back_index -= 1
                continue
            if looks_name_like(previous):
                back_index -= 1
                continue
            if title_stop_line(previous):
                break
            pieces.append(previous)
            back_index -= 1
        if pieces:
            candidate = clean_title_candidate(" ".join(reversed(pieces)))
            candidates.append(
                (candidate_quality(candidate) + 1.2 + max(len(pieces) - 1, 0) * 0.6, candidate)
            )

    for index, line in enumerate(limited_lines[:10]):
        if text_skip_line(line) or title_stop_line(line) or is_continuation_line(line):
            continue
        base_score = candidate_quality(line)
        if base_score < 1.0:
            continue
        joined = [line]
        for next_line in limited_lines[index + 1 : index + 4]:
            if text_skip_line(next_line):
                break
            if title_stop_line(next_line):
                break
            if candidate_quality(next_line) < 0.0:
                break
            if len(" ".join(joined + [next_line])) > 180:
                break
            joined.append(next_line)
        text_candidate = clean_title_candidate(" ".join(joined))
        candidates.append(
            (
                candidate_quality(text_candidate) + min(len(joined) - 1, 2) * 0.4 - index * 0.12,
                text_candidate,
            )
        )

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, candidate = candidates[0]
    if score < 2.0:
        return None, 0.0
    confidence = min(0.88, 0.32 + score / 11.0)
    return candidate or None, confidence


def extract_pdf_title(path: Path) -> tuple[Optional[str], Optional[str], float]:
    layout_candidate, layout_confidence = extract_title_from_layout(path)
    text_candidate, text_confidence = extract_title_from_text(path)
    options = []
    if layout_candidate:
        options.append(("layout", layout_candidate, layout_confidence, candidate_quality(layout_candidate)))
    if text_candidate:
        options.append(("text", text_candidate, text_confidence, candidate_quality(text_candidate)))
    if options:
        method, candidate, confidence, _ = max(options, key=lambda item: (item[3], item[2]))
        return candidate, method, confidence
    return None, None, 0.0


def extract_text_fingerprint(path: Path) -> Optional[str]:
    try:
        text = extract_text(str(path), maxpages=3) or ""
    except Exception:
        return None
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()
    if len(normalized) < 80:
        return None
    truncated = normalized[:12000]
    return hashlib.sha1(truncated.encode("utf-8")).hexdigest()


def score_categories(info_text: str) -> dict[str, int]:
    text = unicodedata.normalize("NFKC", info_text).casefold()
    scores = {category: 0 for category in CATEGORY_KEYWORDS}
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword, weight in keywords.items():
            if unicodedata.normalize("NFKC", keyword).casefold() in text:
                scores[category] += weight
    if "ファクター" in info_text or "factor" in text:
        scores["投資"] -= 2
        scores["統計"] -= 1
    return scores


def predict_folder(current_folder: str, title: Optional[str], stem: str) -> tuple[Optional[str], int, Optional[str], dict[str, int]]:
    source = " ".join(filter(None, [title, stem]))
    scores = score_categories(source)
    predicted, predicted_score = max(scores.items(), key=lambda item: item[1])
    current_score = scores.get(current_folder, 0)
    if predicted == current_folder:
        return predicted, predicted_score, None, scores
    if predicted_score < 4:
        return predicted, predicted_score, None, scores
    if predicted_score < current_score + 3:
        return predicted, predicted_score, None, scores
    reason = f"{predicted} score {predicted_score} vs current {current_score}"
    return predicted, predicted_score, reason, scores


def build_file_info(path: Path) -> FileInfo:
    title, method, confidence = extract_pdf_title(path)
    title = truncate_title(title) if title else None
    title = sanitize_filename(title) if title else None
    text_fingerprint = extract_text_fingerprint(path)
    predicted_folder, predicted_score, move_reason, scores = predict_folder(
        path.parent.name, title, path.stem
    )
    if move_reason and predicted_folder not in MOVE_ALLOWED_TARGETS.get(path.parent.name, set()):
        move_reason = None
    desired_stem = None
    if (
        title
        and path.parent.name in RENAME_ALLOWED_FOLDERS
        and is_generic_stem(path.stem)
        and title_looks_usable(title)
        and confidence >= 0.7
    ):
        desired_stem = title
    return FileInfo(
        path=str(path),
        folder=path.parent.name,
        extension=path.suffix.lower(),
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
        sha256=hash_file(path),
        current_stem=path.stem,
        extracted_title=title,
        title_method=method,
        title_confidence=round(confidence, 3),
        normalized_title=normalize_key(title or path.stem),
        text_fingerprint=text_fingerprint,
        category_scores=scores,
        predicted_folder=predicted_folder,
        predicted_score=predicted_score,
        move_reason=move_reason,
        desired_stem=desired_stem,
    )


def choose_keep(files: list[FileInfo]) -> FileInfo:
    def keep_key(info: FileInfo) -> tuple[int, int, int, float, str]:
        predicted_match = int(bool(info.predicted_folder) and info.predicted_folder == info.folder)
        current_name_match = int(normalize_key(info.current_stem) == normalize_key(info.extracted_title))
        return (
            predicted_match,
            current_name_match,
            info.size,
            info.mtime,
            info.path,
        )

    return max(files, key=keep_key)


def collect_files(root: Path) -> list[FileInfo]:
    files: list[FileInfo] = []
    for folder in FOLDERS:
        folder_path = root / folder
        if not folder_path.exists():
            continue
        for path in sorted(folder_path.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in PDF_EXTENSIONS:
                continue
            if path.name.startswith(".") or path.name.startswith("~$"):
                continue
            files.append(build_file_info(path))
    return files


def build_duplicate_groups(files: list[FileInfo]) -> list[dict]:
    groups: list[dict] = []
    seen_groups: set[tuple[str, ...]] = set()

    by_hash: dict[str, list[FileInfo]] = defaultdict(list)
    for info in files:
        by_hash[info.sha256].append(info)
    for file_group in by_hash.values():
        if len(file_group) < 2:
            continue
        keep = choose_keep(file_group)
        key = tuple(sorted(item.path for item in file_group))
        seen_groups.add(key)
        groups.append(
            {
                "reason": "exact_hash",
                "keep": keep.path,
                "duplicates": [item.path for item in file_group if item.path != keep.path],
            }
        )

    by_text: dict[tuple[str, str], list[FileInfo]] = defaultdict(list)
    for info in files:
        if info.normalized_title and info.text_fingerprint:
            by_text[(info.normalized_title, info.text_fingerprint)].append(info)
    for file_group in by_text.values():
        if len(file_group) < 2:
            continue
        key = tuple(sorted(item.path for item in file_group))
        if key in seen_groups:
            continue
        keep = choose_keep(file_group)
        groups.append(
            {
                "reason": "title_and_text",
                "keep": keep.path,
                "duplicates": [item.path for item in file_group if item.path != keep.path],
            }
        )
    return groups


def unique_target_path(path: Path) -> Path:
    if not path.exists():
        return path
    base = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{base} ({counter}){suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def build_plan(root: Path) -> dict:
    files = collect_files(root)
    duplicates = build_duplicate_groups(files)
    duplicate_paths = {
        duplicate
        for group in duplicates
        for duplicate in group["duplicates"]
    }

    actions: list[dict] = []
    for info in files:
        if info.path in duplicate_paths:
            continue
        final_folder = info.folder
        if info.move_reason and info.predicted_folder:
            final_folder = info.predicted_folder
        final_stem = info.current_stem
        if info.desired_stem and normalize_key(info.desired_stem) != normalize_key(info.current_stem):
            final_stem = info.desired_stem
        target_path = root / final_folder / f"{final_stem}{info.extension}"
        if str(target_path) != info.path:
            actions.append(
                {
                    "type": "move_or_rename",
                    "source": info.path,
                    "target": str(target_path),
                    "title_confidence": info.title_confidence,
                    "title_method": info.title_method,
                    "move_reason": info.move_reason,
                }
            )

    rename_candidates = [
        {
            "path": info.path,
            "current_stem": info.current_stem,
            "desired_stem": info.desired_stem,
            "title_confidence": info.title_confidence,
            "title_method": info.title_method,
        }
        for info in files
        if info.desired_stem and normalize_key(info.desired_stem) != normalize_key(info.current_stem)
    ]
    move_candidates = [
        {
            "path": info.path,
            "from_folder": info.folder,
            "to_folder": info.predicted_folder,
            "reason": info.move_reason,
            "scores": info.category_scores,
        }
        for info in files
        if info.move_reason and info.predicted_folder
    ]

    return {
        "root": str(root),
        "scanned_pdf_count": len(files),
        "rename_candidates": rename_candidates,
        "move_candidates": move_candidates,
        "duplicate_groups": duplicates,
        "actions": actions,
        "files": [asdict(info) for info in files],
    }


def apply_plan(plan: dict) -> dict:
    duplicate_groups = plan.get("duplicate_groups", [])
    actions = plan.get("actions", [])
    log: dict[str, list[dict]] = {
        "deleted": [],
        "moved_or_renamed": [],
        "skipped": [],
    }

    for group in duplicate_groups:
        keep_path = Path(group["keep"])
        keep_hash = hash_file(keep_path) if keep_path.exists() else None
        for duplicate in group["duplicates"]:
            duplicate_path = Path(duplicate)
            if not duplicate_path.exists():
                log["skipped"].append({"path": duplicate, "reason": "missing_before_delete"})
                continue
            if keep_hash and hash_file(duplicate_path) != keep_hash and group["reason"] == "exact_hash":
                log["skipped"].append({"path": duplicate, "reason": "hash_changed"})
                continue
            duplicate_path.unlink()
            log["deleted"].append(
                {"path": duplicate, "kept": group["keep"], "reason": group["reason"]}
            )

    for action in actions:
        source = Path(action["source"])
        target = Path(action["target"])
        if not source.exists():
            log["skipped"].append({"path": action["source"], "reason": "missing_before_move"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        final_target = target
        if target.exists() and source.resolve() != target.resolve():
            if hash_file(source) == hash_file(target):
                source.unlink()
                log["deleted"].append(
                    {
                        "path": str(source),
                        "kept": str(target),
                        "reason": "target_already_exists_same_content",
                    }
                )
                continue
            final_target = unique_target_path(target)
        shutil.move(str(source), str(final_target))
        log["moved_or_renamed"].append(
            {
                "source": str(source),
                "target": str(final_target),
                "move_reason": action.get("move_reason"),
                "title_method": action.get("title_method"),
                "title_confidence": action.get("title_confidence"),
            }
        )

    return log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--log")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    plan = build_plan(root)
    report_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.apply:
        return 0

    if not args.log:
        raise SystemExit("--log is required with --apply")
    log_path = Path(args.log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = apply_plan(plan)
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
