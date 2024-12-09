import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def feature_importance(df, target):
    """Calcula e plota a importância das variáveis."""
    X = df.drop(columns=[target])
    y = df[target]
    model = RandomForestRegressor()
    model.fit(X, y)

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis')
    plt.title('Importância das Variáveis')
    plt.show()

    return importance
