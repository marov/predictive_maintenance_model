import joblib
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from h1st.model.ml_model import MLModel
from h1st.model.ml_modeler import MLModeler

class XGBClassifierModel(MLModel):
    def predict(self, data: dict) -> dict:
        """Run model inference on incoming data and return predictions in a
           dictionary.
           Input data should have key 'X' with data values
        """
        labelEncoder, model = self.base_model
        predictions = labelEncoder.inverse_transform(model.predict(data['X']))
        return {'predictions': predictions}
    def persist(self, version=0) -> str:
        return joblib.dump(self.base_model, f'xgb_classifier_{version}.pkl')
    def load(self, path: str) -> None:
        self.base_model = joblib.load(path)

class XGBClassifierModeler(MLModeler):
    def __init__(self):
        super().__init__()
        self.model_class = XGBClassifierModel

    def train_base_model(self, prepared_data):
        """trains and returns the base ML model that will be wrapped by the
           H1st MyMLModel
        """
        X, y = prepared_data['X_train'], prepared_data['y_train']
        labelEncoder = LabelEncoder()
        y = labelEncoder.fit_transform(y)
        model = XGBClassifier(random_state=42)
        model.fit(X, y)
        return labelEncoder, model

    def load_data(self):
        """Implementing this function is optional, alternatively data can
           be passed directly to the build_model function. If implemented,
           the build_model function can be run without any input.
        """
        pass

    def  evaluate_model(self, data: dict, ml_model: MLModel) -> dict:
        """Optional, if implemented then metrics will be attached to the
           trained model created by the build_model method, and can be
           persisted along with the model
        """
        x_test = {'X': data['X_test']}
        y_test = data['y_test']
        y_pred = ml_model.predict(x_test)['predictions']
        accuracy = accuracy_score(y_test, y_pred)
        cf = confusion_matrix(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        f1 = 2 * precision * recall / (precision + recall)
        return {'accuracy_score': accuracy,
                'confusion_matrix': cf,
                'recall': recall,
                'precision': precision,
                'f1': f1}