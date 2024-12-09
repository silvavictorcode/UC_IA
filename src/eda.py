import seaborn as sns
import matplotlib.pyplot as plt

def plot_distribution(df, column):
    """Plota a distribuição de uma coluna."""
    sns.histplot(df[column], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribuição do {column.capitalize()}')
    plt.xlabel(column.capitalize())
    plt.ylabel('Frequência')
    plt.show()

def plot_correlation(df, target):
    """Plota a correlação entre as variáveis e a variável alvo."""
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix[[target]].sort_values(by=target, ascending=False), annot=True, cmap='coolwarm')
    plt.title(f'Correlação com {target.capitalize()}')
    plt.show()
