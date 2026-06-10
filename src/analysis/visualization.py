import os
import matplotlib.pyplot as plt


def save_bar_chart(
    df,
    x_col,
    y_col,
    title,
    output_path,
    top_n=20,
    xlabel=None,
    ylabel=None
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plot_df = df.head(top_n).copy()
    plot_df = plot_df.sort_values(y_col, ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[x_col], plot_df[y_col])
    plt.title(title)
    plt.xlabel(xlabel or y_col)
    plt.ylabel(ylabel or x_col)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()