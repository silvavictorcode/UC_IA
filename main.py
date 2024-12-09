import data_processing as dp
import eda
import model_training as mt
import feature_analysis as fa
import visualization as vis

def main():
    # Caminhos dos arquivos
    input_path = 'data/cereal.csv'
    processed_path = 'data/cereal_processed.csv'
    target = 'rating'

    # 1. Carregar e pré-processar os dados
    df = dp.load_data(input_path)
    df = dp.preprocess_data(df)
    dp.save_processed_data(df, processed_path)

    # 2. Análise exploratória
    eda.plot_distribution(df, target)
    eda.plot_correlation(df, target)

    # 3. Treinar e comparar modelos
    results = mt.train_and_compare_models(df, target)

    # 4. Análise de importância das variáveis
    importance = fa.feature_importance(df, target)

    # 5. Visualizar resultados
    vis.plot_model_comparisons(results)

if __name__ == "__main__":
    main()
