"""
Journal partition / indexing-tier lookup.

Maps normalised journal names to their partition strings:
  - "SCI Q1" … "SCI Q4"   – Web of Science JCR quartiles
  - "EI"                   – Engineering Index (Compendex)
  - "ESCI"                 – Emerging Sources Citation Index
  - "CSCD"                 – Chinese Science Citation Database
  - "北大核心"              – Peking University Core Journals
  - "无分区"               – No recognised partition

The map covers ~120 widely-cited journals.  For journals not in the
map the function returns ``None`` so callers can display it as
"未知" / "Unknown".
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Static partition map  (key = lower-cased journal name, value = partition)
# ---------------------------------------------------------------------------

JOURNAL_PARTITION_MAP: dict[str, str] = {
    # -------- Nature family --------
    "nature": "SCI Q1",
    "nature medicine": "SCI Q1",
    "nature biotechnology": "SCI Q1",
    "nature methods": "SCI Q1",
    "nature communications": "SCI Q1",
    "nature reviews": "SCI Q1",
    "nature reviews cancer": "SCI Q1",
    "nature reviews genetics": "SCI Q1",
    "nature machine intelligence": "SCI Q1",
    "nature electronics": "SCI Q1",
    "nature energy": "SCI Q1",
    "nature nanotechnology": "SCI Q1",
    "nature chemistry": "SCI Q1",
    "nature neuroscience": "SCI Q1",
    "nature climate change": "SCI Q1",
    "nature human behaviour": "SCI Q1",
    "nature computational science": "SCI Q1",
    # -------- Science family --------
    "science": "SCI Q1",
    "science advances": "SCI Q1",
    "science translational medicine": "SCI Q1",
    # -------- Cell family --------
    "cell": "SCI Q1",
    "cell reports": "SCI Q1",
    "cell systems": "SCI Q1",
    "cancer cell": "SCI Q1",
    "immunity": "SCI Q1",
    "developmental cell": "SCI Q1",
    "molecular cell": "SCI Q1",
    # -------- High-impact interdisciplinary --------
    "the lancet": "SCI Q1",
    "the new england journal of medicine": "SCI Q1",
    "jama": "SCI Q1",
    "bmj": "SCI Q1",
    "plos medicine": "SCI Q1",
    "plos biology": "SCI Q1",
    "plos one": "SCI Q2",
    "elife": "SCI Q1",
    "pnas": "SCI Q1",
    "proceedings of the national academy of sciences": "SCI Q1",
    # -------- Computer Science / AI / ML --------
    "journal of machine learning research": "SCI Q1",
    "jmlr": "SCI Q1",
    "artificial intelligence": "SCI Q1",
    "ieee transactions on neural networks and learning systems": "SCI Q1",
    "ieee transactions on pattern analysis and machine intelligence": "SCI Q1",
    "ieee transactions on image processing": "SCI Q1",
    "ieee transactions on knowledge and data engineering": "SCI Q1",
    "ieee transactions on cybernetics": "SCI Q1",
    "ieee transactions on intelligent transportation systems": "SCI Q1",
    "ieee transactions on information forensics and security": "SCI Q1",
    "acm computing surveys": "SCI Q1",
    "information sciences": "SCI Q1",
    "expert systems with applications": "SCI Q2",
    "knowledge-based systems": "SCI Q1",
    "pattern recognition": "SCI Q1",
    "neural networks": "SCI Q1",
    "neurocomputing": "SCI Q2",
    "computers & security": "SCI Q2",
    # -------- Engineering --------
    "ieee transactions on power electronics": "SCI Q1",
    "ieee transactions on industrial electronics": "SCI Q1",
    "applied energy": "SCI Q1",
    "energy and environmental science": "SCI Q1",
    "energy conversion and management": "SCI Q1",
    "renewable energy": "SCI Q2",
    "renewable and sustainable energy reviews": "SCI Q1",
    "international journal of hydrogen energy": "SCI Q2",
    "journal of cleaner production": "SCI Q1",
    "chemical engineering journal": "SCI Q1",
    # -------- Medicine --------
    "annals of oncology": "SCI Q1",
    "gut": "SCI Q1",
    "journal of hepatology": "SCI Q1",
    "journal of clinical oncology": "SCI Q1",
    "circulation": "SCI Q1",
    "european heart journal": "SCI Q1",
    "journal of the american college of cardiology": "SCI Q1",
    "radiology": "SCI Q1",
    "chest": "SCI Q2",
    "journal of internal medicine": "SCI Q2",
    "medicine": "SCI Q3",
    # -------- Materials / Chemistry --------
    "advanced materials": "SCI Q1",
    "acs nano": "SCI Q1",
    "nano letters": "SCI Q1",
    "nano energy": "SCI Q1",
    "journal of the american chemical society": "SCI Q1",
    "angewandte chemie": "SCI Q1",
    "chemistry of materials": "SCI Q1",
    "npj 2d materials and applications": "SCI Q1",
    # -------- Environmental --------
    "environmental science & technology": "SCI Q1",
    "water research": "SCI Q1",
    "science of the total environment": "SCI Q1",
    "environmental pollution": "SCI Q2",
    "chemosphere": "SCI Q2",
    # -------- Mathematics / Physics --------
    "physical review letters": "SCI Q1",
    "physical review b": "SCI Q1",
    "journal of mathematical analysis and applications": "SCI Q2",
    "applied mathematics and computation": "SCI Q2",
    "mathematics of computation": "SCI Q1",
    # -------- EI-primary journals --------
    "ieee access": "EI",
    "ieee communications letters": "EI",
    "journal of systems architecture": "EI",
    "microprocessors and microsystems": "EI",
    "simulation modelling practice and theory": "EI",
    "computers & electrical engineering": "EI",
    # -------- Chinese core journals --------
    "中国科学": "CSCD",
    "科学通报": "CSCD",
    "计算机学报": "CSCD",
    "软件学报": "CSCD",
    "自动化学报": "CSCD",
    "电子学报": "CSCD",
    "中国机械工程": "北大核心",
    "化工学报": "CSCD",
    "高等学校化学学报": "CSCD",
    "中华医学杂志": "北大核心",
}


def lookup_partition(journal_name: str | None) -> str | None:
    """
    Return the partition string for *journal_name*, or ``None`` if unknown.

    Matching is case-insensitive and strips leading/trailing whitespace.
    """
    if not journal_name:
        return None
    key = journal_name.lower().strip()
    return JOURNAL_PARTITION_MAP.get(key)
