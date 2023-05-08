from predictive_maintenance.modeler import PredictiveMaintananceModeler

if __name__ == '__main__':
    modeler = PredictiveMaintananceModeler()
    model = modeler.build_model()
    model.persist('1st_version')
    accuracy = model.metrics['accuracy_score'] * 100
    print(f"Accuracy (test): {accuracy:.1f}%")
    precision_score_value = model.metrics['precision'] * 100
    print(f"Precision (test): {precision_score_value:.1f}%")
    recall_score_value = model.metrics['recall'] * 100
    print(f"Recall (test): {recall_score_value:.1f}%")
    f1_score_value = model.metrics['f1'] * 100
    print(f"F1 (test): {f1_score_value:.1f}%")
