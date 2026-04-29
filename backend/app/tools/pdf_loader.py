"""
MedicalAI — tools/pdf_loader.py
PDF 文档加载与文本分块工具。
"""

from typing import List

from langchain_core.documents import Document

from app.core.logging_config import logger


def load_pdf(pdf_path: str) -> List[Document]:
    """使用 PyPDFLoader 加载 PDF 文件的全部页面。"""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    logger.info("已从 PDF 加载 %d 页：%s", len(docs), pdf_path)
    return docs


def _preprocess_medical_text(text: str) -> str:
    """
    预处理医学文本：识别列举式结构，在编号前插入段落分隔符。

    问题根因
    --------
    医学书中"临床表现：1.发热 2.咳嗽 3.胸痛"被句号切成残缺片段。
    在分块前把编号前插入 \n\n，让 splitter 在完整语义边界处断开。
    """
    import re

    # 1. 句末标点后跟数字编号：在编号前插入段落分隔
    #    匹配"。1." "？2、" "！（3）"等格式
    text = re.sub(r'(?<=[\u3002\uff01\uff1f\n])(\s*)(\d{1,2}[.\u3001\uff09])', r'\n\n\2', text)
    text = re.sub(r'(?<=[\u3002\uff01\uff1f\n])(\s*)[\uff08(](\d{1,2})[\uff09)]', r'\n\n(\2)', text)

    # 2. 中文序号 ①②③... 前插入分隔
    for ch in '\u2460\u2461\u2462\u2463\u2464\u2465\u2466\u2467\u2468\u2469':
        text = text.replace(ch, '\n\n' + ch)

    # 3. 章节标题：第X章/节/条 前插入分隔
    text = re.sub(
        r'(?<=[\u3002\uff01\uff1f\n])(\u7b2c[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\d]+[\u7ae0\u8282\u6761\u6b3e])',
        r'\n\n\1', text
    )

    # 4. 清理多余连续空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def split_documents(docs: List[Document]) -> List[Document]:
    """
    将文档切分为带重叠的文本块。

    改造说明（相比原版）
    -------------------
    原版问题：chunk_size=512，RecursiveCharacterTextSplitter 按句号切割，
    遇到"症状：1.发热 2.咳嗽 3.头痛"会把完整症状清单切成残缺片段。

    改造方案：
    1. 分块前对文本做预处理（_preprocess_medical_text），
       在列举编号前插入 \n\n，引导 splitter 在语义完整的边界处断开。
    2. chunk_size 从 512 → 768：医学段落往往包含多个症状/诊断要点，
       512字符经常在列举中途截断；768字符约500中文字，能容纳完整列举。
    3. chunk_overlap 从 64 → 128：增大重叠避免跨块语义丢失，
       尤其对"以上症状如持续超过3天"这类承接上文的表述。
    4. 添加 add_start_index=True，在 metadata 里记录原文位置，
       供调试和可观测性使用。

    权衡说明
    --------
    chunk_size 增大会导致向量库文档数从3904减少到约2600，
    但每个 chunk 的信息完整性更高，是信息密度 vs 召回粒度的 tradeoff。
    医疗场景里"信息完整"比"召回粒度细"更重要。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 预处理：在列举结构处插入段落分隔，引导分块在语义边界断开
    preprocessed = []
    for doc in docs:
        processed_text = _preprocess_medical_text(doc.page_content)
        preprocessed.append(Document(
            page_content=processed_text,
            metadata=doc.metadata,
        ))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=768,           # 原512 → 768：容纳完整列举式内容
        chunk_overlap=128,        # 原64 → 128：增大重叠避免跨块语义丢失
        length_function=len,
        add_start_index=True,     # 新增：在 metadata 记录原文位置
        separators=[
            "\n\n",             # 段落分隔（最优先，预处理已在列举处插入）
            "\n",
            "。", "！", "？",
            "；", "……",
            ". ", "! ", "? ",
            "，", ",",
            " ", "",
        ],
        is_separator_regex=False,
    )
    splits = splitter.split_documents(preprocessed)
    logger.info(
        "文档已切分为 %d 个文本块（医疗优化分块：size=768，overlap=128，预处理列举结构）",
        len(splits),
    )
    return splits


def process_pdf(pdf_path: str) -> List[Document]:
    """加载 PDF 并切分文本块的便捷封装函数。"""
    return split_documents(load_pdf(pdf_path))