# ============================================================
# main.py'ye EKLENECEK KISIMLAR
# ============================================================
# 
# 1. Model yükleme kısmını şununla değiştir:
# 
#    with open("best_model_xgboost_shap.pkl", "rb") as f:
#        package = pickle.load(f)
#    MODEL         = package['model']
#    FEATURE_COLS  = package['feature_cols']
#    THRESHOLD     = package['threshold']
#    SHAP_EXPLAINER = package.get('shap_explainer', None)
#
# 2. Aşağıdaki endpoint'i main.py'ye ekle:
# ============================================================

import shap
import numpy as np

# Global değişken olarak ekle (model yükleme kısmında)
# SHAP_EXPLAINER = package.get('shap_explainer', None)


@app.post("/predict/explain", summary="SHAP açıklamalı tekil tahmin")
def predict_with_explanation(
    req: PredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if SHAP_EXPLAINER is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer yüklenmedi. best_model_xgboost_shap.pkl kullanın."
        )

    # Özellik hesapla
    features = calculate_features(req.smiles.strip())
    if features is None:
        raise HTTPException(status_code=400, detail="Geçersiz SMILES formatı")

    df = pd.DataFrame([features])
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLS].values

    # Tahmin
    proba       = MODEL.predict_proba(X)[0]
    prob_active = float(proba[1])
    prediction  = "active" if prob_active >= THRESHOLD else "inactive"

    # SHAP değerleri
    shap_values = SHAP_EXPLAINER.shap_values(X)
    sv = shap_values[0] if not isinstance(shap_values, list) else shap_values[0]

    # Top 15 özellik
    feature_importance = sorted(
        zip(FEATURE_COLS, sv.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_features = [
        {
            "feature":    feat,
            "shap_value": round(val, 5),
            "direction":  "active" if val > 0 else "inactive"
        }
        for feat, val in feature_importance[:15]
    ]

    # Sadece Lipinski tanımlayıcıları
    lipinski_shap = [
        {
            "feature":    feat,
            "shap_value": round(val, 5),
            "direction":  "active" if val > 0 else "inactive"
        }
        for feat, val in feature_importance
        if not feat.startswith("fp_")
    ]

    # DB'ye kaydet
    pred = Prediction(
        user_id=current_user.id,
        smiles=req.smiles.strip(),
        prediction=prediction,
        prob_active=round(prob_active, 4),
        prob_inactive=round(float(proba[0]), 4),
        confidence=round(max(proba) * 100, 2)
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    return {
        "smiles":          req.smiles,
        "prediction":      prediction,
        "prob_active":     round(prob_active, 4),
        "prob_inactive":   round(float(proba[0]), 4),
        "confidence":      round(max(proba) * 100, 2),
        "saved_id":        pred.id,
        "shap": {
            "expected_value": round(float(SHAP_EXPLAINER.expected_value)
                               if not isinstance(SHAP_EXPLAINER.expected_value, np.ndarray)
                               else float(SHAP_EXPLAINER.expected_value[1]), 5),
            "top_features":   top_features,
            "lipinski_shap":  lipinski_shap
        }
    }