import pandas as pd
import urllib.request
from pathlib import Path
from src.core.context import PipelineContext

class IngestionStep:
    def __init__(self, context: PipelineContext):
        self.context = context
        self.logger = context.logger
        self.config = context.load_config("data")["ingestion"]

    def run(self):
        self.logger.info("Etapa de Ingestão")
        
        raw_url = self.config["raw_data_source"]
        local_raw_path = self.context.get_path(self.config["local_raw_path"])
        output_parquet_path = self.context.get_path(self.config["output_parquet_path"])
        
        local_raw_path.parent.mkdir(parents=True, exist_ok=True)
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not local_raw_path.exists():
            self.logger.info(f"Download dos dados brutos: {raw_url}")
            urllib.request.urlretrieve(raw_url, local_raw_path)
            self.logger.info("Download concluído")
        else:
            self.logger.info("MSG ERRO: Ficheiro CSV bruto já existe localmente")
            
        df = pd.read_csv(local_raw_path)
        
        self.logger.info(f"Gravando os dados em formato Parquet: {output_parquet_path}")
        df.to_parquet(output_parquet_path, index=False)
        
        self.logger.info(f"Resulto final, ingestão finalizada! Total de linhas: {len(df)} | Colunas: {len(df.columns)}")