from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def train_model(X_train, y_train, model):
    """Treina o modelo fornecido."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_test, y_pred):
    """Calcula MSE e R² do modelo."""
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

def train_and_compare_models(df, target):
    """Treina e compara vários modelos."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Linear Regression": LinearRegression()
    }

    results = {}
    for name, model in models.items():
        model = train_model(X_train, y_train, model)
        y_pred = model.predict(X_test)
        results[name] = evaluate_model(y_test, y_pred)
    
    return results
