"""
修复后的数据获取 - 使用500.com HTML表格解析和CWL API
"""
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from pathlib import Path
from typing import Optional

from utils.helpers import get_logger


def fetch_from_500_html(cfg, max_draws=200):
    """Parse lottery data from 500.com HTML table (both DLT and SSQ)."""
    base_url = cfg.data_sources.get("500com", 
        f"https://datachart.500.com/{cfg.short}/history/newinc/history.php")
    url = f"{base_url}?start=&end=&limit={max_draws}"
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, headers=headers, timeout=30)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    
    rows = []
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        texts = [td.get_text(strip=True) for td in tds]
        if len(texts) >= 1 + cfg.main_count + cfg.sub_count and re.match(r"^\d{5}$", texts[0]):
            try:
                period = int(texts[0])
                if not (25000 <= period <= 27000):
                    continue  # filter valid period range
                row = {"period": period}
                for i in range(cfg.main_count):
                    row[cfg.main_cols[i]] = int(texts[1 + i])
                for i in range(cfg.sub_count):
                    row[cfg.sub_cols[i]] = int(texts[1 + cfg.main_count + i])
                rows.append(row)
            except (ValueError, IndexError):
                continue
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["period"])
        df = df.sort_values("period", ascending=False).reset_index(drop=True)
    return df


def fetch_ssq_cwl(max_draws=500):
    """Fetch SSQ from CWL API (more data than 500.com)."""
    url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    params = {"name": "ssq", "issueCount": max_draws}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://www.cwl.gov.cn/",
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    data = r.json()
    if data.get("state") != 0:
        raise ValueError(f"CWL API error: {data.get('message')}")
    
    def cwl_period_to_short(fp):
        return int(fp[-5:]) if len(fp) >= 5 else int(fp)
    
    rows = []
    for draw in data.get("result", []):
        period = cwl_period_to_short(draw.get("code", "0"))
        nums = [int(x) for x in draw.get("red", "").split(",") if x.strip()]
        nums += [int(x) for x in draw.get("blue", "").split(",") if x.strip()]
        if len(nums) < 7:
            continue
        row = {"period": period}
        for i in range(6):
            row[f"red_{i+1}"] = nums[i]
        row["blue_1"] = nums[6]
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["period"])
        df = df.sort_values("period", ascending=False).reset_index(drop=True)
    return df


def update_data(cfg, force_refresh=False, source="auto", max_draws=None):
    """Main update function - replaces the broken CSV-based one."""
    logger = get_logger(cfg)
    csv_path = Path(cfg.history_csv)
    
    # Load from cache unless force refresh
    if not force_refresh and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            required = ["period"] + cfg.main_cols + cfg.sub_cols
            if all(c in df.columns for c in required):
                df = df.drop_duplicates(subset=["period"])
                df = df.sort_values("period", ascending=False).reset_index(drop=True)
                logger.info(f"Loaded {len(df)} {cfg.name} draws from cache")
                return df
        except Exception:
            pass
    
    logger.info(f"Fetching {cfg.name}...")
    
    # For SSQ: prefer CWL API (more data)
    if cfg.short == "ssq":
        try:
            df = fetch_ssq_cwl(max_draws or 500)
            if not df.empty:
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                logger.info(f"SSQ: {len(df)} draws from CWL API")
                return df
        except Exception as e:
            logger.warning(f"CWL API failed: {e}")
    
    # Fallback: 500.com HTML
    df = fetch_from_500_html(cfg, max_draws or 200)
    if not df.empty:
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"{cfg.name}: {len(df)} draws from 500.com")
        return df
    
    logger.error(f"Failed to fetch {cfg.name}")
    return pd.DataFrame()
