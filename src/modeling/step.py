import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.core.context import PipelineContext

class ModelingStep:
    def __init__(self, context: PipelineContext):
        self.context = context
        self.logger = context.logger
        self.config = context.load_config("modeling")["modeling"]

    def run(self):
        self.logger.info("START")
        
        input_path = self.context.get_path(self.config["input_parquet_path"])
        df = pd.read_parquet(input_path)
        
        target = self.config["target_column"]
        X = df.drop(columns=[target])
        y = df[target]

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config["test_size"], 
            random_state=self.config["random_state"],
            stratify=y
        )

        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, num_cols),
                ('cat', cat_pipeline, cat_cols)
            ]
        )

        pipeline_steps = [('preprocessor', preprocessor)]
        
        if self.config.get("use_pca", False):
            self.logger.info(f"PCA ativado {self.config['pca_components']} componentes.")
            pipeline_steps.append(
                ('pca', PCA(n_components=self.config["pca_components"], random_state=self.config["random_state"]))
            )
            run_name = "RandomForest_PCA_Tuned"
        else:
            run_name = "RandomForest_Baseline_Tuned"

        pipeline_steps.append(
            ('classifier', RandomForestClassifier(random_state=self.config["random_state"], n_jobs=-1))
        )

        pipeline = Pipeline(steps=pipeline_steps)

        mlflow_uri = str(self.context.get_path(self.config["mlflow_tracking_uri"].replace("sqlite:///", "")))
        mlflow.set_tracking_uri(f"sqlite:///{mlflow_uri}")
        mlflow.set_experiment(self.config["experiment_name"])

        with mlflow.start_run(run_name=run_name):
            tuning_cfg = self.config.get("hyperparameter_tuning", {})
            
            if tuning_cfg.get("enable", False):
                search = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=tuning_cfg["param_distributions"],
                    n_iter=tuning_cfg["n_iter"],
                    cv=tuning_cfg["cv_folds"],
                    scoring=tuning_cfg["scoring"],
                    random_state=self.config["random_state"],
                    n_jobs=1
                )
                search.fit(X_train, y_train)
                modelo_final = search.best_estimator_
                
                self.logger.info(f"Parâmetros encontrados: {search.best_params_}")
                mlflow.log_params(search.best_params_)
            else:
                self.logger.info("Sem tuning")
                pipeline.fit(X_train, y_train)
                modelo_final = pipeline

            y_pred = modelo_final.predict(X_test)
            y_proba = modelo_final.predict_proba(X_test)[:, 1]
            
            metrics = {
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }
            
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(modelo_final, "model")

            self.logger.info("Modeling criado")