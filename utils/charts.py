"""
Reusable, premium Plotly visualization components for AgriPredict AI.
Implements standardized aesthetics, graceful empty-state handling,
and responsive formatting across agricultural charts.
"""

from typing import List, Optional, Any, Dict
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config.settings import THEME, PLOTLY_COLORS


def empty_figure(message: str = "No data available for the selected filters.") -> go.Figure:
    """
    Renders an informative empty state figure when data is missing or filtered out.
    """
    fig = go.Figure()
    fig.add_annotation(
        text=f"🍃 {message}",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=15, color=THEME["text_muted"], family="Plus Jakarta Sans, sans-serif"),
        align="center"
    )
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(244, 247, 244, 0.4)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=350
    )
    return fig


def apply_theme_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """
    Applies standard AgriPredict AI visual theme and typography to any Plotly figure.
    """
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=16, color=THEME["primary"]),
            x=0.01,
            y=0.96
        ),
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color=THEME["text_dark"]),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=30, t=55, b=50),
        height=height,
        hoverlabel=dict(
            bgcolor=THEME["primary"],
            font_size=12,
            font_family="Plus Jakarta Sans, sans-serif",
            font_color="white"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        tickfont=dict(size=11)
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#E2E8F0",
        linecolor="#CBD5E1",
        tickfont=dict(size=11)
    )
    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    orientation: str = "v",
    text_auto: bool = False,
    height: int = 420
) -> go.Figure:
    """
    Generates a themed bar chart with interactive hovers and clean formatting.
    """
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return empty_figure(f"Data not available for {title or 'Bar Chart'}")

    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        orientation=orientation,
        text_auto=text_auto,
        color_discrete_sequence=PLOTLY_COLORS
    )
    return apply_theme_layout(fig, title=title, height=height)


def create_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: str = "",
    markers: bool = True,
    height: int = 420
) -> go.Figure:
    """
    Generates a themed line chart for time-series or sequential trends.
    """
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return empty_figure(f"Data not available for {title or 'Line Chart'}")

    fig = px.line(
        df,
        x=x,
        y=y,
        color=color,
        markers=markers,
        color_discrete_sequence=PLOTLY_COLORS
    )
    return apply_theme_layout(fig, title=title, height=height)


def create_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    title: str = "",
    trendline: Optional[str] = None,
    height: int = 440
) -> go.Figure:
    """
    Generates a themed scatter plot with optional regression trendline.
    """
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return empty_figure(f"Data not available for {title or 'Scatter Plot'}")

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        hover_data=hover_data,
        trendline=trendline,
        color_discrete_sequence=PLOTLY_COLORS
    )
    return apply_theme_layout(fig, title=title, height=height)


def create_box_plot(
    df: pd.DataFrame,
    x: Optional[str],
    y: str,
    color: Optional[str] = None,
    title: str = "",
    height: int = 420
) -> go.Figure:
    """
    Generates a themed box plot for distribution analysis and outlier detection.
    """
    if df is None or df.empty or y not in df.columns:
        return empty_figure(f"Data not available for {title or 'Box Plot'}")

    fig = px.box(
        df,
        x=x,
        y=y,
        color=color,
        color_discrete_sequence=PLOTLY_COLORS,
        points="outliers"
    )
    return apply_theme_layout(fig, title=title, height=height)


def create_histogram(
    df: pd.DataFrame,
    x: str,
    color: Optional[str] = None,
    nbins: int = 30,
    marginal: Optional[str] = "box",
    title: str = "",
    height: int = 420
) -> go.Figure:
    """
    Generates a themed histogram with marginal distribution overlays.
    """
    if df is None or df.empty or x not in df.columns:
        return empty_figure(f"Data not available for {title or 'Histogram'}")

    fig = px.histogram(
        df,
        x=x,
        color=color,
        nbins=nbins,
        marginal=marginal,
        color_discrete_sequence=PLOTLY_COLORS,
        opacity=0.85
    )
    return apply_theme_layout(fig, title=title, height=height)


def create_donut_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str = "",
    height: int = 420
) -> go.Figure:
    """
    Generates a modern donut pie chart for categorical distributions.
    """
    if df is None or df.empty or names not in df.columns or values not in df.columns:
        return empty_figure(f"Data not available for {title or 'Donut Chart'}")

    fig = px.pie(
        df,
        names=names,
        values=values,
        hole=0.45,
        color_discrete_sequence=PLOTLY_COLORS
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return apply_theme_layout(fig, title=title, height=height)


def create_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
    height: int = 450
) -> go.Figure:
    """
    Generates a correlation heatmap with customized green-amber divergence.
    """
    if df is None or df.empty:
        return empty_figure("Cannot generate correlation heatmap on empty dataset.")

    target_cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    if len(target_cols) < 2:
        return empty_figure("At least 2 numeric features are required for correlation analysis.")

    corr = df[target_cols].corr().round(2)

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[
            [0.0, "#D32F2F"],      # Negative correlation: Red
            [0.5, "#F8FAFC"],      # Neutral: Off-white
            [1.0, "#1B4D3E"]       # Positive correlation: Deep Forest Green
        ],
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        textfont={"size": 11, "family": "Plus Jakarta Sans"},
        hoverongaps=False
    ))
    return apply_theme_layout(fig, title=title, height=height)


def create_confusion_matrix_chart(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Model Confusion Matrix",
    height: int = 440
) -> go.Figure:
    """
    Visualizes a confusion matrix with clear actual vs predicted labels and color graduation.
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred: {c}" for c in class_names],
        y=[f"Act: {c}" for c in class_names],
        colorscale=[
            [0.0, "#FFFFFF"],
            [1.0, "#1B4D3E"]
        ],
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 13, "family": "Plus Jakarta Sans"},
        hoverongaps=False
    ))
    return apply_theme_layout(fig, title=title, height=height)


def create_feature_importance_chart(
    features: List[str],
    importances: List[float],
    title: str = "Model Feature Importance",
    top_n: int = 15,
    height: int = 420
) -> go.Figure:
    """
    Generates a ranked horizontal bar chart of feature importances.
    """
    df_imp = pd.DataFrame({"Feature": features, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=True).tail(top_n)

    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation="h",
        marker_color=THEME["primary_light"],
        text=[f"{v:.3f}" for v in df_imp["Importance"]],
        textposition="outside"
    ))
    return apply_theme_layout(fig, title=title, height=height)


def create_radar_chart(
    categories: List[str],
    values: List[float],
    title: str = "Soil Nutrient Profile",
    fill: str = "toself",
    height: int = 400
) -> go.Figure:
    """
    Generates a radar/spider chart for multi-dimensional soil nutrient balance.
    """
    # Close the radar loop
    cats = categories + [categories[0]]
    vals = values + [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=vals,
        theta=cats,
        fill=fill,
        fillcolor="rgba(46, 125, 50, 0.25)",
        line=dict(color=THEME["primary_light"], width=2)
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) * 1.15 if values else 100]),
            bgcolor="#F9FBF9"
        ),
        showlegend=False
    )
    return apply_theme_layout(fig, title=title, height=height)
