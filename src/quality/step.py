import pandas as pd
import json
from datetime import datetime
from src.core.context import PipelineContext

class QualityStep:
    def __init__(self, context: PipelineContext):
        self.context = context
        self.logger = context.logger
        self.config = context.load_config("quality")["quality"]

    def run(self):
        self.logger.info("A iniciar validação de Qualidade de Dados...")
        
        input_path = self.context.get_path(self.config["input_parquet_path"])
        report_path = self.context.get_path(self.config["report_output_path"])
        
        df = pd.read_parquet(input_path)
        report = {"timestamp": datetime.now().isoformat(), "results": [], "status": "PASSED"}
        
        for exp in self.config["expectations"]:
            col = exp["column"]
            exp_type = exp["type"]
            passed = True
            details = ""

            if exp_type == "not_null":
                null_count = df[col].isnull().sum()
                passed = null_count == 0
                details = f"Encontrados {null_count} nulos."
            
            elif exp_type == "values_in_set":
                invalid_count = (~df[col].isin(exp["values"])).sum()
                passed = invalid_count == 0
                details = f"Encontrados {invalid_count} valores fora de {exp['values']}."
                
            elif exp_type == "min_value":
                invalid_count = (df[col] < exp["value"]).sum()
                passed = invalid_count == 0
                details = f"Encontrados {invalid_count} valores menores que {exp['value']}."

            report["results"].append({
                "column": col,
                "expectation": exp_type,
                "passed": bool(passed),
                "details": details
            })

            if not passed:
                report["status"] = "FAILED"
                self.logger.error(f"Falha na validação: Coluna '{col}' | Regra: {exp_type} | {details}")
            else:
                self.logger.info(f"Sucesso: Coluna '{col}' cumpriu a regra '{exp_type}'.")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
            
        if report["status"] == "FAILED":
            self.logger.warning(f"Qualidade de dados comprometida. Relatório gerado em: {report_path}")
        else:
            self.logger.info(f"Todas as verificações de qualidade passaram! Relatório salvo em: {report_path}")