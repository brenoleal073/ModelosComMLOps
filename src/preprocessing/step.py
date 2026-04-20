import pandas as pd
from src.core.context import PipelineContext

class PreprocessingStep:
    def __init__(self, context: PipelineContext):
        self.context = context
        self.logger = context.logger
        self.config = context.load_config("preprocessing")["preprocessing"]

    def run(self):
        self.logger.info("Iniciar Pré-processamento")
        
        input_path = self.context.get_path(self.config["input_parquet_path"])
        output_path = self.context.get_path(self.config["output_parquet_path"])
        
        df = pd.read_parquet(input_path)
        
        cols_to_drop = self.config.get("drop_columns", [])
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        self.logger.info(f"Colunas removidas {cols_to_drop}")
        
        num_cols = self.config.get("numerical_columns", [])
        for col in num_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                
        cat_cols = self.config.get("categorical_columns", [])
        for col in cat_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Preprocessing concluído")
        self.logger.info(f"Dados guardados em: {output_path}")