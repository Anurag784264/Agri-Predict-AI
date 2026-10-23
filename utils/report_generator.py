"""
Analytical report generation pipeline for AgriPredict AI.
Compiles executive PDF summaries using ReportLab and generates CSV metrics export packages.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import io
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)

from config.settings import REPORTS_DIR


def generate_executive_pdf_report(
    kpi_data: Dict[str, Any],
    dataset_status: Dict[str, Any],
    ml_metrics: Optional[List[Dict[str, Any]]] = None,
    output_filename: Optional[str] = None
) -> Tuple[Path, bytes]:
    """
    Constructs a comprehensive, publication-grade analytical PDF report
    documenting data statistics, findings, and model evaluations.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"AgriPredict_Analytics_Report_{timestamp}.pdf"

    pdf_path = REPORTS_DIR / output_filename
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom ReportLab Typography
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1B4D3E"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "ReportSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#556B2F"),
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1B4D3E"),
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1A2E1A"),
        spaceAfter=8
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("AgriPredict AI — Agricultural Analytics & ML Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | Third-Year AI & Data Science Project",
        subtitle_style
    ))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2E7D32"), spaceAfter=14))

    # Executive Summary & KPIs
    elements.append(Paragraph("1. Executive Summary & Platform Indicators", h2_style))
    elements.append(Paragraph(
        "AgriPredict AI integrates multi-source agricultural datasets spanning crop production statistics, "
        "hourly multi-city meteorological records, laboratory soil nutrient profiles, chemical fertilizer classifications, "
        "and national agricultural market mandi price distributions into a unified analytical framework.",
        body_style
    ))

    kpi_table_data = [
        ["Metric Indicator", "Observed Value", "Scope & Context"],
        ["Total Platform Records", str(kpi_data.get("total_records_formatted", "N/A")), "Across all loaded data pipelines"],
        ["Distinct Agricultural Crops", str(kpi_data.get("crop_count", "N/A")), "Documented across Indian states & districts"],
        ["Total Recorded Production", str(kpi_data.get("total_production_formatted", "N/A")), "National cumulative production records"],
        ["Total Cultivated Area", str(kpi_data.get("total_area_formatted", "N/A")), "Reported agricultural area"],
        ["Average National Yield", f"{kpi_data.get('avg_yield', 'N/A')} Tonnes/Ha", "Production divided by cultivated area"],
        ["Market Commodities", str(kpi_data.get("commodity_count", "N/A")), "APMC mandi cross-sectional price records"],
        ["Weather Observation Hubs", f"{kpi_data.get('weather_cities', 8)} Metropolitan Hubs", "Hourly meteorological records (2009-2020)"]
    ]

    t_kpi = Table(kpi_table_data, colWidths=[160, 130, 240])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B4D3E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F9FBF9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F4F7F4")]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5)
    ]))
    elements.append(t_kpi)
    elements.append(Spacer(1, 14))

    # Dataset Verification Status
    elements.append(Paragraph("2. Dataset Verification & Ingestion Status", h2_style))
    ds_table_data = [["Dataset Name", "Status", "Source Path"]]
    for key, info in dataset_status.items():
        st_text = "Verified" if info.get("found") else "Missing"
        path_text = Path(info.get("path", "")).name if info.get("path") else "Not configured"
        ds_table_data.append([info.get("title", key), st_text, path_text])

    t_ds = Table(ds_table_data, colWidths=[180, 80, 270])
    t_ds.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F4F7F4")]),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4)
    ]))
    elements.append(t_ds)
    elements.append(Spacer(1, 14))

    # Machine Learning Evaluation Summary
    elements.append(Paragraph("3. Machine Learning Models Evaluation", h2_style))
    if ml_metrics and len(ml_metrics) > 0:
        ml_table_data = [["Task", "Algorithm", "Primary Metric", "Score", "Validation Protocol"]]
        for m in ml_metrics:
            ml_table_data.append([
                m.get("task", "Classification"),
                m.get("algorithm", "Model"),
                m.get("primary_metric", "Accuracy"),
                str(m.get("score", "-")),
                m.get("protocol", "80/20 Train-Test Split")
            ])
        t_ml = Table(ml_table_data, colWidths=[120, 130, 90, 70, 120])
        t_ml.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B4D3E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F4F7F4")]),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 1), (-1, -1), 8.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4)
        ]))
        elements.append(t_ml)
    else:
        elements.append(Paragraph(
            "Machine learning benchmarking models can be trained and evaluated via the ML Model Lab page. "
            "Evaluations dynamically populate this section upon execution.",
            body_style
        ))
    elements.append(Spacer(1, 14))

    # Limitations & Academic Disclaimers
    elements.append(Paragraph("4. Analytical Limitations & Domain Disclaimers", h2_style))
    elements.append(Paragraph(
        "• <b>Market Prices:</b> Represents an empirical cross-sectional snapshot across 122 agricultural commodities. "
        "Due to the absence of continuous longitudinal date timestamps, market forecasting was deliberately omitted to prevent spurious projections.<br/>"
        "• <b>Weather-District Integration:</b> Meteorological data originates from 8 urban/metropolitan weather stations. "
        "Data integration with district crop production is restricted to verified geographical overlaps and overlapping years (2009-2015).<br/>"
        "• <b>Agricultural Advisory:</b> Predictions from soil and fertilizer models provide decision support only and do not replace "
        "official field soil tests conducted by certified agronomists.",
        body_style
    ))

    # Build PDF document
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    # Save to disk
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    return pdf_path, pdf_bytes
