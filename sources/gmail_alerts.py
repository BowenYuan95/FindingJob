#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gmail_alerts.py — 用 Gmail API 读取 JobAlerts 标签的提醒邮件,
                  用本地 LLM (Qwen3.5) 把邮件正文抽成结构化职位(正则作降级备份)。

前置:
  1. Google Cloud:启用 Gmail API、建 OAuth 桌面客户端,下载 credentials.json 放本目录。
  2. Gmail:建 JobAlerts 标签 + 过滤器,把 Seek/Indeed/LinkedIn 提醒邮件打上该标签。
  3. pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib requests
  4. LM Studio 开着、加载 Qwen3.5(LLM 抽取需要;没开则自动退回正则)。

单独测试:  python gmail_alerts.py
接回主程序:from gmail_alerts import fetch_gmail_alerts
"""

import os
import re
import json
import base64
import logging
from html.parser import HTMLParser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import LLM_MODEL
from infrastructure.lmstudio import LM_CLIENT

logger = logging.getLogger(__name__)

# ---- Gmail ----
SCOPES        = ["https://www.googleapis.com/auth/gmail.readonly"]
LABEL_NAME    = "JobAlerts"     # 与你 Gmail 里的标签名一致(大小写敏感!)
MAX_DAYS_OLD  = 7
CREDS_FILE    = "credentials.json"
TOKEN_FILE    = "token.json"

# ---- LLM 抽取(LM Studio)----
USE_LLM_EXTRACT = True
DEBUG_RAW       = False       # 调试:打印 LLM 原始返回
BODY_MAXLEN     = 10000                 # 压缩 URL 后的正文上限(更安全)


# ================================================================
# OAuth
# ================================================================

def _get_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


# ================================================================
# 取正文
# ================================================================

class _Strip(HTMLParser):
    def __init__(self): super().__init__(); self.out = []
    def handle_data(self, d): self.out.append(d)
    def text(self): return " ".join(self.out)

def _decode(data):
    return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")

def _extract_body(payload):
    text_plain, text_html = "", ""
    def walk(part):
        nonlocal text_plain, text_html
        mime = part.get("mimeType", ""); data = part.get("body", {}).get("data")
        if mime == "text/plain" and data:  text_plain += _decode(data)
        elif mime == "text/html" and data: text_html += _decode(data)
        for sub in part.get("parts", []) or []: walk(sub)
    walk(payload)
    if text_plain.strip():
        return re.sub(r"\s+", " ", text_plain)
    if text_html.strip():
        p = _Strip(); p.feed(text_html); return re.sub(r"\s+", " ", p.text())
    return ""


# ================================================================
# 解析:LLM 抽取(主)+ 正则(降级)
# ================================================================

def _extract_json_array(txt):
    # 去掉 <think> 标签式思考
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL)
    txt = re.sub(r"```json|```", "", txt)
    # 模型常在前面写一大段 "Thinking Process:" 等思考文字,
    # 我们不管前言,直接定位 JSON 数组的起止。
    start = txt.find("[")
    if start == -1:
        return "[]"
    end = txt.rfind("]")
    if end > start:                      # 有完整收尾 ]
        return txt[start:end + 1]
    last_obj = txt.rfind("}")            # 被截断:补 ]
    if last_obj > start:
        return txt[start:last_obj + 1] + "]"
    return "[]"

def _denoise(body):
    """砍掉页脚噪音,并用可逆占位符压缩超长 URL。"""
    url_map = {}

    def replace_url(match):
        token = f"[URL_{len(url_map) + 1}]"
        url_map[token] = match.group(0)
        return token

    body = re.sub(r"https?://\S{80,}", replace_url, body)
    # 2) 砍掉页脚/退订/推荐区块
    cuts = [
        "unsubscribe", "Unsubscribe", "退订", "manage your alerts",
        "Manage alerts", "You are receiving", "This email was sent",
        "Why you are receiving", "Update your settings", "Get the app",
        "Download the app", "Follow us", "Privacy Policy", "© ",
    ]
    cut_at = len(body)
    for c in cuts:
        idx = body.find(c)
        if idx != -1:
            cut_at = min(cut_at, idx)
    return body[:cut_at].strip() or body, url_map


def _restore_url(value, url_map):
    raw = str(value or "").strip()
    if raw in url_map:
        return url_map[raw]
    match = re.fullmatch(r"\[*URL_(\d+)\]*", raw, flags=re.I)
    if match:
        return url_map.get(f"[URL_{match.group(1)}]", "")
    return raw


def _llm_extract(body, src):
    """把一封邮件正文交给 Qwen3.5,抽出职位数组。失败返回 None(触发降级)。"""
    denoised, url_map = _denoise(body)
    prompt = f"""Extract ALL job listings from this {src} alert email.
Output ONLY a JSON array. Your FIRST character must be '['. No thinking, no preamble, no explanation.
Each item: {{"title":"", "company":"", "location":"", "url":"", "desc":""}}
For "desc": a SHORT one-line summary from any context in the email (salary, work type,
key skills if shown). If nothing available, use "".
Skip footers/ads/unsubscribe. If none, output [].

EMAIL:
{denoised[:BODY_MAXLEN]}

JSON array:"""
    try:
        response = LM_CLIENT.chat_completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON extractor. Output only a JSON array starting with [. Never explain or think."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            timeout=240,
            max_tokens=8000,
        )
        msg_obj = response["choices"][0]["message"]
        txt = msg_obj.get("content") or ""
        if not txt.strip():                       # content 空则尝试 reasoning 字段
            txt = msg_obj.get("reasoning_content") or msg_obj.get("reasoning") or ""
        finish = response["choices"][0].get("finish_reason", "")
        if finish == "length":
            logger.warning(f"[gmail]     ⚠ 输出被截断(finish_reason=length)")
        # 调试:打印原始返回的开头,看模型到底在写什么
        if DEBUG_RAW:
            logger.info(f"[gmail]     原始返回前300字: {txt[:300]!r}")
            logger.info(f"[gmail]     原始返回总长: {len(txt)} 字")
        arr = json.loads(_extract_json_array(txt))
        out = []
        for it in arr:
            title = (it.get("title") or "").strip()
            if not title:
                continue
            out.append({
                "title": title,
                "company": (it.get("company") or "").strip(),
                "location": (it.get("location") or "").strip(),
                "url": _restore_url(it.get("url"), url_map),
                "description": (it.get("desc") or "").strip(),
                "source": src, "salary": "", "created": "",
            })
        return out
    except Exception as e:
        logger.warning(f"[gmail]   LLM 抽取失败({src}),退回正则: {e}")
        return None

def _regex_extract(body, src):
    """降级方案:粗匹配 '职位 at 公司'。"""
    out = []
    for m in re.finditer(r"([A-Z][\w &/\-]{6,70})\s+at\s+([\w &/\-,.]{2,50})", body):
        out.append({
            "title": m.group(1).strip(), "company": m.group(2).strip(),
            "location": "", "url": "", "description": body[:1500],
            "source": src, "salary": "", "created": "",
        })
    return out


# ================================================================
# 主函数
# ================================================================

def fetch_gmail_alerts():
    try:
        svc = _get_service()
    except Exception as e:
        logger.error(f"[gmail] 初始化失败(检查 credentials.json): {e}")
        return []

    query = f"label:{LABEL_NAME} newer_than:{MAX_DAYS_OLD}d"
    msg_ids = []
    page_token = None
    try:
        while True:
            kwargs = {"userId": "me", "q": query, "maxResults": 100}
            if page_token:
                kwargs["pageToken"] = page_token
            resp = svc.users().messages().list(**kwargs).execute(num_retries=3)
            msg_ids.extend(resp.get("messages", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    except Exception as e:
        logger.warning(f"[gmail] 列邮件失败: {e}")
        return []
    logger.info(f"[gmail] 命中 {len(msg_ids)} 封 {LABEL_NAME} 邮件(近 {MAX_DAYS_OLD} 天)")

    jobs = []
    for m in msg_ids:
        try:
            msg = svc.users().messages().get(
                userId="me", id=m["id"], format="full"
            ).execute(num_retries=3)
        except Exception as e:
            logger.warning(f"[gmail] 读取邮件 {m.get('id', '?')} 失败,跳过: {e}")
            continue
        headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
        sender = headers.get("from", "").lower()
        src = ("seek"     if "seek"     in sender else
               "indeed"   if "indeed"   in sender else
               "linkedin" if "linkedin" in sender else "email")
        subject = headers.get("subject", "")[:60]
        body = _extract_body(msg["payload"])
        logger.info(f"[gmail]   · {src:8s} | 正文 {len(body)} 字 | {subject}")

        parsed = None
        if USE_LLM_EXTRACT:
            parsed = _llm_extract(body, src)
        if parsed is None:                      # LLM 关闭或失败 -> 正则降级
            parsed = _regex_extract(body, src)

        jobs.extend(parsed)
        logger.info(f"[gmail]   {src:8s} 解析出 {len(parsed)} 条")

    logger.info(f"[gmail] 合计解析 {len(jobs)} 条")
    return jobs


if __name__ == "__main__":
    for j in fetch_gmail_alerts():
        loc = f" — {j['location']}" if j['location'] else ""
        print(f"  - [{j['source']}] {j['title']}  @ {j['company']}{loc}")
