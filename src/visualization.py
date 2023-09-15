import plotly.graph_objects as go

import preprocessing


def get_radar_chart(input_data):
    input_data = preprocessing.get_scaled_values(input_data)
    categories = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave points",
        "symmetry",
        "fractal_dimension",
    ]

    fig = go.Figure()

    for category in ["mean", "se", "worst"]:
        columns = [f"{col}_{category}" for col in categories]
        values = [
            input_data[col] for col in columns
        ]  # Get slider values from input_data dictionary
        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=f"{category.capitalize()} Value",
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True
    )
    return fig
