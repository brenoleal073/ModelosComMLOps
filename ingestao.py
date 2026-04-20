import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.context import PipelineContext
from src.ingestion.step import IngestionStep

if __name__ == "__main__":
    context = PipelineContext.from_notebook(__file__)
    
    step = IngestionStep(context)
    step.run()