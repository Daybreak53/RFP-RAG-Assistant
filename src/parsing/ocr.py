import pdfplumber
import unicodedata
import os
import json
import csv
import numpy as np

from pathlib import Path
from paddleocr import PaddleOCR
from hwp_hwpx_parser import Reader
from func_timeout import func_timeout, FunctionTimedOut

from src.parsing.meta_db import load_metadata_db
from src.parsing.parser import create_chunks, convert_chunks_to_rag_format

class SimpleDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def OCR_parsing(
        documents,
        data_dir="data",
        csv_path="data/data_list.csv",
        match_threshold=0.55,
        **kwargs
):
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='korean', device='gpu')

    # 1. 기존 메타데이터 DB 로드 (하위 호환성 유지)
    metadata_map, _ = load_metadata_db(
        csv_path=Path(csv_path),
        data_dir=Path(data_dir),
        threshold=match_threshold,
    )

    print(f"[OCR Parser] Processing {len(documents)} documents...")

    # ======= 문서 파일 스캔 및 텍스트 추출 =======
    processed_docs = []
    processed_files = set()

    for doc in documents:
        file_path = doc.metadata.get("source")

        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, os.path.basename(file_path))

        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        if file_path in processed_files:
            continue

        try:
            # --- TRACK A: PDF 처리 (pdfplumber 활용 및 표 구조 보존) ---
            if ext == ".pdf":
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    full_table_text = ""

                    for i, page in enumerate(pdf.pages, 1):
                        text = page.extract_text() or ""
                        tables = page.extract_tables()

                        full_text += text + "\n"
                        # 테이블 텍스트 누적
                        # 마크다운 표 형식으로 변환 (헤더와 본문 구분선 추가)
                        table_text = ""
                        for t in tables:
                            for row_idx, r in enumerate(t):
                                # 빈칸(None)은 '-' 로 치환하고 마크다운 파이프(|)로 감싸기
                                row_str = "| " + " | ".join(
                                    [str(c).strip().replace('\n', ' ') if c else "-" for c in r]) + " |"
                                table_text += row_str + "\n"
                                # 첫 번째 행(헤더) 밑에 구분선(---|---) 추가
                                if row_idx == 0:
                                    table_text += "|" + "|".join(["---"] * len(r)) + "|\n"
                            table_text += "\n"

                        full_table_text += table_text

                        # OCR 처리 (텍스트가 너무 적을 때만 수행)
                        if len(text.strip()) < 50:
                            img = page.to_image(resolution=200).original  # 해상도 100으로 낮춰서 속도 확보
                            img = np.array(img)
                            res = ocr_engine.ocr(img)
                            if res and res[0]:
                                # OCR 처리 (텍스트가 너무 적을 때만 수행)
                                ocr_texts = []
                                try:
                                    # 1. 안전한 데이터만 필터링: line이 리스트고, 좌표(line[0])도 정상적인 형태일 때만 추출
                                    valid_lines = [
                                        line for line in res[0]
                                        if isinstance(line, list) and len(line) == 2
                                           and isinstance(line[0], list) and len(line[0]) > 0
                                    ]

                                    # 2. 정렬 (y좌표 기준)
                                    sorted_res = sorted(valid_lines,
                                                        key=lambda x: (x[0][0][1] // 10, x[0][0][0]))

                                    # 3. 텍스트 추출
                                    for line in sorted_res:
                                        text_data = line[1]
                                        if isinstance(text_data, (tuple, list)) and len(text_data) > 0:
                                            ocr_texts.append(str(text_data[0]))
                                        elif isinstance(text_data, str):
                                            ocr_texts.append(text_data)

                                except Exception as e:
                                    for line in res[0]:
                                        try:
                                            if isinstance(line, list) and len(line) > 1:
                                                if isinstance(line[1], (tuple, list)) and len(line[1]) > 0:
                                                    ocr_texts.append(str(line[1][0]))
                                                else:
                                                    ocr_texts.append(str(line[1]))
                                        except:
                                            continue

                                ocr_refined_text = " ".join(ocr_texts).strip()
                                if ocr_refined_text:
                                    full_text += f"\n[OCR 추출 텍스트]\n{ocr_refined_text}\n"

                    processed_docs.append(SimpleDocument(
                        f"{full_text}\n\n[Table]\n{full_table_text}",
                        {"source": file_path, "file_name": file_name}
                    ))

            # --- TRACK B: HWP 처리 (타임아웃 및 안전 추출 적용) ---
            elif ext in [".hwp", ".hwpx"]:
                def _parse(fp):
                    with Reader(fp) as r: return f"{r.text}\n\n[Table]\n{str(r.tables)}"

                content = func_timeout(120, _parse, args=[file_path])
                processed_docs.append(SimpleDocument(content, {"source": file_path, "file_name": file_name, "page": 1}))

            processed_files.add(file_path)

        except FunctionTimedOut:
            print(f"[Warning] 600s (10m) Timeout: {file_name}", flush=True)
            processed_files.add(file_path)
            continue

        except Exception as e:
            print(f"[Error] Parsing {file_name}: {e}")

    # 2. 청킹 및 RAG 포맷 변환
    chunks = create_chunks(documents=processed_docs, **kwargs)
    rag_data = convert_chunks_to_rag_format(chunks, metadata_map=metadata_map)

    # 3. 메타데이터 매핑 및 정제
    raw_csv_rows = []
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_csv_rows.append(row)
    except Exception as csv_err:
        print(f"[Warning] Failed to load CSV with utf-8-sig ({csv_err}). Retrying with 'euc-kr'...")
        with open(csv_path, "r", encoding="euc-kr") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_csv_rows.append(row)

    file_col, budget_col, org_col, date_col = None, None, None, None
    start_date_col, end_date_col = None, None

    if raw_csv_rows:
        headers = list(raw_csv_rows[0].keys())
        clean_headers = [str(h).strip().replace(" ", "") for h in headers]
        print(f"[Debug] Detected raw CSV headers from file: {headers}")

        for h, ch in zip(headers, clean_headers):
            if ch in ['파일명', 'file_name', 'filename']: file_col = h; break
        if not file_col:
            for h, ch in zip(headers, clean_headers):
                if '파일' in ch or 'file' in ch: file_col = h; break

        for h, ch in zip(headers, clean_headers):
            if ch in ['사업금액', '사업비', '예산', 'budget', 'budget_amt', '금액']: budget_col = h; break
        if not budget_col:
            for h, ch in zip(headers, clean_headers):
                if '금액' in ch or '예산' in ch or '비' in h or 'budget' in ch: budget_col = h; break

        for h, ch in zip(headers, clean_headers):
            if ch in ['발주기관', '주관기관', '수요기관', 'organization', 'org']: org_col = h; break
        if not org_col:
            for h, ch in zip(headers, clean_headers):
                if '기관' in ch or 'org' in ch: org_col = h; break

        for h, ch in zip(headers, clean_headers):
            if '공개' in ch or '공고' in ch or 'date' in ch or '등록' in ch: date_col = h; break
        for h, ch in zip(headers, clean_headers):
            if '시작' in ch or 'start' in ch: start_date_col = h; break
        for h, ch in zip(headers, clean_headers):
            if '마감' in ch or 'deadline' in ch or '종료' in ch: end_date_col = h; break

        print(f"[Debug] Dynamic mapping results -> File: [{file_col}], Budget: [{budget_col}], Org: [{org_col}]")

    def clean_text(text):
        if not text:
            return ""
        normalized = unicodedata.normalize('NFC', str(text)).lower()
        for ext_to_del in ['.pdf', '.hwp', '.hwpx', '.docx', '.xlsx', '.csv']:
            normalized = normalized.replace(ext_to_del, "")
        return normalized.replace("_", "").replace(" ", "").replace("…", "").strip()

    # 4. 메타데이터 정밀 매칭 보정
    match_success_count = 0
    debug_count = 0

    for chunk in rag_data:
        is_nested = "metadata" in chunk and isinstance(chunk["metadata"], dict)
        target_dict = chunk["metadata"] if is_nested else chunk

        chunk_file = target_dict.get("file_name", "") or target_dict.get("title", "") or target_dict.get("doc_id",
                                                                                                         "") or chunk.get(
            "id", "")
        clean_chunk_key = clean_text(chunk_file)

        matched_meta = None
        if clean_chunk_key:
            for row in raw_csv_rows:
                csv_file_name = row.get(file_col, "") if file_col else ""
                clean_csv_key = clean_text(csv_file_name)

                if clean_csv_key:
                    if clean_csv_key in clean_chunk_key or clean_chunk_key in clean_csv_key:
                        matched_meta = row
                        match_success_count += 1
                        if debug_count < 3:
                            print(
                                f"[Match Success] Chunk Keyword [{clean_chunk_key}] <-> CSV Filename [{clean_csv_key}]")
                            debug_count += 1
                        break

        # 5. 실제 복구된 데이터 내부 계층에 int 정수형으로 변환 적용
        if matched_meta:
            real_budget = matched_meta.get(budget_col, "0") if budget_col else "0"
            try:
                real_budget = str(real_budget).replace(",", "").replace("원", "").replace(" ", "").strip()
                target_dict["budget"] = int(float(real_budget)) if real_budget else 0
            except:
                target_dict["budget"] = 0

            if date_col: target_dict["announcement_date"] = matched_meta.get(date_col, "")
            if start_date_col: target_dict["bid_start"] = matched_meta.get(start_date_col, "")
            if end_date_col: target_dict["bid_deadline"] = matched_meta.get(end_date_col, "")
            if org_col: target_dict["organization"] = matched_meta.get(org_col,
                                                                       target_dict.get("organization", "기타 기관"))

            target_dict["title"] = unicodedata.normalize('NFC', str(chunk_file))
            target_dict["file_name"] = unicodedata.normalize('NFC', str(chunk_file))

            fn_lower = str(target_dict.get("file_name", "")).lower()
            if ".hwp" in fn_lower or ".hwpx" in fn_lower:
                target_dict["file_type"] = "hwp"
            elif ".pdf" in fn_lower:
                target_dict["file_type"] = "pdf"

    print(f"[Info] Successfully matched {match_success_count} chunks out of {len(rag_data)} with CSV records.")

    # 6. 최종 보정 완료된 데이터를 캐시 파일로 저장
    cache_path = os.path.join(data_dir, "parsed_cache_ocr.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=4)
        print(f"[Cache Complete] Successfully saved corrected RAG data to '{cache_path}'")
    except Exception as cache_err:
        print(f"[Error] Failed to write cache file: {cache_err}")

    print(f"[Info] Parsing completed. Total {len(rag_data)} chunks generated.")
    return rag_data
