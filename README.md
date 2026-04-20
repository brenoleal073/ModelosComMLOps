import os

readme_content = """# Hotel Booking MLOps - Pipeline de Previsão de Cancelamento

Este repositório contém uma solução completa de Engenharia de Machine Learning para a predição de cancelamentos de reservas hoteleiras. O projeto foi estruturado seguindo os princípios de MLOps, garantindo modularidade, rastreabilidade e prontidão para produção.

## 🚀 Arquitetura do Projeto

O sistema foi desenhado separando **Política** (configurações YAML) de **Mecanismo** (motores em Python). A esteira de dados é composta por:

1.  **Ingestão:** Conversão de dados brutos para formato Parquet por performance.
2.  **Qualidade:** Validação de contratos de dados (nulos, tipos e limites).
3.  **Pré-processamento:** Limpeza cirúrgica e prevenção de *Data Leakage*.
4.  **Modelagem:** Treinamento rastreável com MLflow e otimização de hiperparâmetros (RandomizedSearchCV).
5.  **Serving:** API de alta performance com FastAPI e documentação Swagger.

## 📁 Estrutura de Pastas

```text
├── config/             # Configurações de cada etapa (YAML)
├── src/                # Motores do pipeline
│   ├── core/           # Gerenciamento de contexto e infra
│   ├── ingestion/      # Lógica de ingestão
│   ├── quality/        # Validação de dados
│   ├── preprocessing/  # Limpeza e features
│   └── modeling/       # Treino e experimentação
├── data/               # Dados (Raw, Processed, Features)
├── mlruns.db           # Banco de dados de experimentos MLflow
├── api.py              # Servidor de inferência (FastAPI)
└── *.py                # Scripts orquestradores (raiz)
