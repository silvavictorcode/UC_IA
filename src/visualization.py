import matplotlib.pyplot as plt

def plot_model_comparisons(results):
    """Plota a comparação de MSE e R² entre os modelos."""
    models = list(results.keys())
    mse = [results[model][0] for model in models]
    r2 = [results[model][1] for model in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(models, mse, color='skyblue')
    axes[0].set_title('Comparação de MSE')
    axes[0].set_ylabel('MSE')

    axes[1].bar(models, r2, color='salmon')
    axes[1].set_title('Comparação de R²')
    axes[1].set_ylabel('R²')

    plt.tight_layout()
    plt.show()
