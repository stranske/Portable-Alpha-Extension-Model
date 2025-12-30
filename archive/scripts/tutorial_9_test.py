import plotly.graph_objects as go

from pa_core.viz.export_bundle import save


# Create two simple figures
def main():
    fig1 = go.Figure(data=go.Bar(x=[1, 2, 3], y=[4, 5, 6]), layout_title_text="Bar Plot")
    fig2 = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[6, 5, 4]), layout_title_text="Scatter Plot")

    save(
        [fig1, fig2],
        "tutorial_9_output/figure",
        alt_texts=["Bar chart showing values", "Scatter plot showing reverse values"],
    )
    print("Saved figures in tutorial_9_output/")


if __name__ == "__main__":
    main()
