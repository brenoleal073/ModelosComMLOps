import logging
from pathlib import Path
import yaml

class PipelineContext:

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.logger = self._setup_logger()
        self._configs = {}

    def _setup_logger(self):
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger("HotelMLOps")

    def load_config(self, config_name: str):
        if config_name not in self._configs:
            config_path = self.root_dir / "config" / f"{config_name}.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"Configuração não encontrada: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                self._configs[config_name] = yaml.safe_load(f)
        return self._configs[config_name]

    def get_path(self, relative_path: str) -> Path:
        return self.root_dir / relative_path

    @classmethod
    def from_notebook(cls, current_file_path: str):
        current_dir = Path(current_file_path).resolve().parent
        while current_dir != current_dir.parent:
            if (current_dir / "requirements.txt").exists():
                return cls(current_dir)
            current_dir = current_dir.parent
        return cls(Path(current_file_path).resolve().parent)