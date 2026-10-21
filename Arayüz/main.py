"""
ALZHEİMER İLAÇ ANALİZ SİSTEMİ - FastAPI Backend
=================================================

Kurulum:
  pip install fastapi uvicorn sqlalchemy passlib[bcrypt] python-jose[cryptography] python-multipart rdkit requests

Çalıştırma:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional, List
import pickle, numpy as np, pandas as pd, io, csv, requests, urllib.parse
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from rdkit.Chem import inchi as rdInchi
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# ============================================================
# AYARLAR
# ============================================================
SECRET_KEY = "alzheimer-drug-system-secret-key-2024"
ALGORITHM  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

DATABASE_URL    = "sqlite:///./alzheimer_system.db"
CHEMBL_API      = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_TIMEOUT  = 15

# ============================================================
# VERİTABANI
# ============================================================
engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


class User(Base):
    __tablename__ = "users"
    id          = Column(Integer, primary_key=True, index=True)
    email       = Column(String, unique=True, index=True, nullable=False)
    full_name   = Column(String, nullable=False)
    hashed_pw   = Column(String, nullable=False)
    role        = Column(String, default="biologist")
    created_at  = Column(DateTime, default=datetime.utcnow)
    predictions = relationship("Prediction", back_populates="user")
    batch_jobs  = relationship("BatchJob",   back_populates="user")


class Prediction(Base):
    __tablename__ = "predictions"
    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False)
    smiles        = Column(Text,    nullable=False)
    prediction    = Column(String,  nullable=False)
    prob_active   = Column(Float,   nullable=False)
    prob_inactive = Column(Float,   nullable=False)
    confidence    = Column(Float,   nullable=False)
    chembl_id     = Column(String,  nullable=True)
    chembl_name   = Column(String,  nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)
    user          = relationship("User", back_populates="predictions")


class BatchJob(Base):
    __tablename__ = "batch_jobs"
    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename       = Column(String)
    total          = Column(Integer, default=0)
    active_count   = Column(Integer, default=0)
    inactive_count = Column(Integer, default=0)
    error_count    = Column(Integer, default=0)
    created_at     = Column(DateTime, default=datetime.utcnow)
    user           = relationship("User", back_populates="batch_jobs")


Base.metadata.create_all(bind=engine)

# ============================================================
# ML MODELİ — TARGET_COLS DÜZELTİLDİ
# ============================================================
print("Model yükleniyor...")
MODEL        = None
FEATURE_COLS = None
TARGET_COLS  = []        # ← YENİ: pipeline'daki target one-hot sütunları
THRESHOLD    = 0.5
_package     = None      # ← ham pickle paketi, TARGET_COLS için saklanıyor

try:
    with open("best_model_xgboost_fixed.pkl", "rb") as f:
        _package = pickle.load(f)
    MODEL        = _package["model"]
    FEATURE_COLS = _package["feature_cols"]
    TARGET_COLS  = _package.get("target_cols", [])   # ← DÜZELTİLDİ
    THRESHOLD    = _package["threshold"]
    print(f"✓ Model yüklendi : {_package['model_name']}")
    print(f"  threshold      : {THRESHOLD}")
    print(f"  feature sayısı : {len(FEATURE_COLS)}")
    print(f"  target_cols    : {len(TARGET_COLS)} adet")
except FileNotFoundError:
    print("⚠  Model dosyası bulunamadı — tahmin devre dışı")

morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


def calculate_features(smiles: str) -> Optional[dict]:
    """SMILES → descriptor + Morgan fp + MACCS keys sözlüğü."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        d = {
            "MW":                 Descriptors.MolWt(mol),
            "LogP":               Descriptors.MolLogP(mol),
            "NumHDonors":         Descriptors.NumHDonors(mol),
            "NumHAcceptors":      Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds":  Descriptors.NumRotatableBonds(mol),
            "AromaticProportion": (len(mol.GetAromaticAtoms()) / mol.GetNumAtoms()
                                   if mol.GetNumAtoms() > 0 else 0),
            "TPSA":               Descriptors.TPSA(mol),
            "NumRings":           Descriptors.RingCount(mol),
            "MolMR":              Descriptors.MolMR(mol),
            "NumAromaticRings":   Descriptors.NumAromaticRings(mol),
            "NumAliphaticRings":  Descriptors.NumAliphaticRings(mol),
            "BertzCT":            Descriptors.BertzCT(mol),
            "HeavyAtomCount":     mol.GetNumHeavyAtoms(),
            "FractionCSP3":       Descriptors.FractionCSP3(mol),
        }
        fp = morgan_gen.GetFingerprint(mol)
        for i, bit in enumerate(np.array(fp)):
            d[f"fp_{i}"] = int(bit)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        for i, bit in enumerate(maccs):
            d[f"maccs_{i}"] = int(bit)
        return d
    except Exception:
        return None


def run_prediction(smiles: str) -> Optional[dict]:
    """
  m
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")

    features = calculate_features(smiles)
    if features is None:
        return None

    df = pd.DataFrame([features])

    # ── Target one-hot sütunlarını sıfır ekle (bilinmeyen hedef) ──────────────
    # Pipeline'da 15 target sütunu vardı (target_CHEMBL*).
    # Tahmin sırasında hangi hedef için analiz yapıldığı bilinmediğinden sıfır.
    for tc in TARGET_COLS:
        if tc not in df.columns:
            df[tc] = 0

    # ── Eksik diğer sütunları sıfırla ─────────────────────────────────────────
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    proba = MODEL.predict_proba(df[FEATURE_COLS].values)[0]
    return {
        "prediction":    "active" if proba[1] >= THRESHOLD else "inactive",
        "prob_active":   round(float(proba[1]), 4),
        "prob_inactive": round(float(proba[0]), 4),
        "confidence":    round(float(max(proba)) * 100, 2),
    }

# ============================================================
# AUTH
# ============================================================
pwd_ctx       = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(pw: str) -> str:
    return pwd_ctx.hash(pw)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    exc = HTTPException(
        status_code=401,
        detail="Geçersiz token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise exc
    except JWTError:
        raise exc
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise exc
    return user

# ============================================================
# PYDANTIC ŞEMALAR
# ============================================================

class RegisterRequest(BaseModel):
    email:     str
    full_name: str
    password:  str
    role:      Optional[str] = "biologist"


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str
    user_name:    str
    role:         str


class PredictRequest(BaseModel):
    smiles: str


class PredictResponse(BaseModel):
    smiles:        str
    prediction:    str
    prob_active:   float
    prob_inactive: float
    confidence:    float
    saved_id:      Optional[int] = None
    chembl_id:     Optional[str] = None
    chembl_name:   Optional[str] = None


class PredictionHistory(BaseModel):
    id:          int
    smiles:      str
    prediction:  str
    prob_active: float
    confidence:  float
    chembl_id:   Optional[str]
    chembl_name: Optional[str]
    created_at:  datetime
    class Config:
        from_attributes = True


class BatchSummary(BaseModel):
    id:             int
    filename:       Optional[str]
    total:          int
    active_count:   int
    inactive_count: int
    error_count:    int
    created_at:     datetime
    class Config:
        from_attributes = True

# ============================================================
# ChEMBL YARDIMCI FONKSİYONLAR
# ============================================================

def _chembl_get(path: str, params: dict) -> Optional[dict]:
    """ChEMBL REST API'ye GET atar; hata varsa None döner."""
    try:
        r = requests.get(
            f"{CHEMBL_API}/{path}",
            params=params,
            timeout=CHEMBL_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"ChEMBL hatası [{path}]: {e}")
        return None


def _smiles_to_inchi_key(smiles: str) -> Optional[str]:
    """SMILES → InChI Key dönüşümü."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        inchi_str = rdInchi.MolToInchi(mol)
        if inchi_str:
            return rdInchi.InchiToInchiKey(inchi_str)
    except Exception as e:
        print(f"InChI Key dönüşüm hatası: {e}")
    return None


def chembl_lookup_by_smiles(smiles: str) -> Optional[dict]:
    """
    SMILES → ChEMBL molecule kaydı.

       1) RDKit canonical SMILES ile exact match
      2) InChI Key (tam eşleşme)
      3) InChI Key connectivity block — max_phase'e göre sıralı  ← DÜZELTİLDİ
      4) Tanimoto benzerlik araması (threshold=99) fallback        ← YENİ

    Memantine gibi ilaçların RDKit canonical SMILES'i ChEMBL'dekiyle
    birebir eşleşmeyebilir; connectivity block + similarity fallback
    bu durumu tolere eder.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    canonical = Chem.MolToSmiles(mol)

    # ── Strateji 1: canonical SMILES exact match ──────────────────────────────
    data = _chembl_get("molecule.json", {
        "molecule_structures__canonical_smiles__exact": canonical,
        "limit": 1,
    })
    if data and data.get("molecules"):
        hit = data["molecules"][0]
        print(f"ChEMBL bulundu (canonical): {hit.get('molecule_chembl_id')}")
        return hit

    # ── Strateji 2: InChI Key tam eşleşme ────────────────────────────────────
    inchi_key = _smiles_to_inchi_key(smiles)
    if inchi_key:
        print(f"InChI Key deneniyor: {inchi_key}")
        data = _chembl_get("molecule.json", {
            "molecule_structures__standard_inchi_key__exact": inchi_key,
            "limit": 1,
        })
        if data and data.get("molecules"):
            hit = data["molecules"][0]
            print(f"ChEMBL bulundu (InChI Key): {hit.get('molecule_chembl_id')}")
            return hit

    # ── Strateji 3: connectivity block (stereo toleranslı) ───────────────────
    # InChI Key formatı: XXXXXXXXXXXXXX-XXXXXXXXXX-X
    # İlk blok yapısal bağlantıyı temsil eder; stereokimya farkını tolere eder.
    # Birden fazla sonuç gelebilir → max_phase'e göre en ileri klinik adayı seç.
    if inchi_key:
        connectivity = inchi_key.split("-")[0]
        print(f"Connectivity block deneniyor: {connectivity}")
        data = _chembl_get("molecule.json", {
            "molecule_structures__standard_inchi_key__startswith": connectivity,
            "limit": 10,   # ← DÜZELTİLDİ: 1 → 10, en iyi adayı seçebilmek için
        })
        if data and data.get("molecules"):
            mols = data["molecules"]
            # max_phase en yüksek olan kaydı tercih et (onaylı ilaç → 4)
            mols.sort(key=lambda x: x.get("max_phase") or -1, reverse=True)
            hit = mols[0]
            print(f"ChEMBL bulundu (connectivity, max_phase={hit.get('max_phase')}): "
                  f"{hit.get('molecule_chembl_id')}")
            return hit

    # ── Strateji 4: Yüksek-eşikli benzerlik araması (fallback) ──────────────
    # Önceki stratejiler başarısız olursa %99 Tanimoto ile dene.
    # Memantine gibi küçük, simetrik moleküllerde canonical SMILES farkı olabilir.
    print(f"Similarity fallback deneniyor (threshold=99): {canonical[:50]}")
    try:
        encoded = urllib.parse.quote(canonical, safe="")
        data = _chembl_get(f"similarity/{encoded}/99.json", {"limit": 5})
        if data and data.get("molecules"):
            mols = data["molecules"]
            mols.sort(key=lambda x: (
                float(x.get("similarity") or 0),
                x.get("max_phase") or -1
            ), reverse=True)
            hit = mols[0]
            sim = hit.get("similarity")
            print(f"ChEMBL bulundu (similarity={sim}): {hit.get('molecule_chembl_id')}")
            return hit
    except Exception as e:
        print(f"Similarity fallback hatası: {e}")

    print(f"ChEMBL'de bulunamadı: {smiles[:60]}")
    return None


def chembl_bioactivities(chembl_id: str, limit: int = 10) -> list:
    """Verilen ChEMBL ID için biyoaktivite kayıtları."""
    data = _chembl_get("activity.json", {
        "molecule_chembl_id":    chembl_id,
        "target_organism":       "Homo sapiens",
        "pchembl_value__isnull": False,
        "limit":                 limit,
        "order_by":              "-pchembl_value",   # ← DÜZELTİLDİ: yüksekten düşüğe
    })
    if not data:
        return []
    activities = []
    for a in data.get("activities", []):
        activities.append({
            "target_pref_name": a.get("target_pref_name"),
            "target_chembl_id": a.get("target_chembl_id"),
            "standard_type":    a.get("standard_type"),
            "standard_value":   a.get("standard_value"),
            "standard_units":   a.get("standard_units"),
            "pchembl_value":    a.get("pchembl_value"),
            "assay_chembl_id":  a.get("assay_chembl_id"),
        })
    return activities


def chembl_similar_molecules(smiles: str, threshold: int = 70, limit: int = 10) -> list:
    """Tanimoto benzerliğine göre ChEMBL'den benzer moleküller."""
    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol) if mol else smiles
    encoded   = urllib.parse.quote(canonical, safe="")
    data      = _chembl_get(f"similarity/{encoded}/{threshold}.json", {"limit": limit})
    if not data:
        return []
    results = []
    for m in data.get("molecules", []):
        props = m.get("molecule_properties") or {}
        results.append({
            "chembl_id":  m.get("molecule_chembl_id"),
            "name":       m.get("pref_name") or "—",
            "max_phase":  m.get("max_phase"),
            "similarity": m.get("similarity"),
            "mw":         props.get("full_mwt"),
            "alogp":      props.get("alogp"),
            "smiles":     (m.get("molecule_structures") or {}).get("canonical_smiles"),
        })
    return results


def chembl_molecule_detail(chembl_id: str) -> Optional[dict]:
    """Tek bir ChEMBL ID için tam detay."""
    data = _chembl_get(f"molecule/{chembl_id}.json", {})
    if not data:
        return None
    m       = data
    props   = m.get("molecule_properties") or {}
    structs = m.get("molecule_structures") or {}
    ro5     = int(props.get("num_ro5_violations") or 0)
    return {
        "chembl_id":       m.get("molecule_chembl_id"),
        "name":            m.get("pref_name") or "—",
        "max_phase":       m.get("max_phase"),
        "formula":         props.get("full_molformula"),
        "mw":              props.get("full_mwt"),
        "alogp":           props.get("alogp"),
        "hba":             props.get("hba"),
        "hbd":             props.get("hbd"),
        "rotatable_bonds": props.get("rtb"),
        "tpsa":            props.get("psa"),
        "ro5_violations":  ro5,
        "ro5_pass":        ro5 == 0,
        "smiles":          structs.get("canonical_smiles"),
        "inchi_key":       structs.get("standard_inchi_key"),
        "molecule_type":   m.get("molecule_type"),
    }

# ============================================================
# FASTAPI UYGULAMASI
# ============================================================
app = FastAPI(
    title="Alzheimer İlaç Analiz Sistemi",
    description=(
        "Biyologlar için molekül biyoaktivite tahmin platformu — ChEMBL entegrasyonlu\n\n"
        "v2.2.0: target_cols fix · ChEMBL 4-strateji lookup · similarity fallback"
    ),
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# AUTH ENDPOINT'LERİ
# ============================================================

@app.post("/auth/register", summary="Yeni biyolog kaydı")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Bu email zaten kayıtlı")
    user = User(
        email=req.email,
        full_name=req.full_name,
        hashed_pw=hash_password(req.password),
        role=req.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": f"{user.full_name} başarıyla kaydedildi", "id": user.id}


@app.post("/auth/login", response_model=TokenResponse, summary="Giriş yap")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username).first()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Email veya şifre hatalı")
    token = create_token({"sub": user.email}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user_name":    user.full_name,
        "role":         user.role,
    }


@app.get("/auth/me", summary="Mevcut kullanıcı bilgisi")
def me(current_user: User = Depends(get_current_user)):
    return {
        "id":         current_user.id,
        "email":      current_user.email,
        "full_name":  current_user.full_name,
        "role":       current_user.role,
        "created_at": current_user.created_at,
    }

# ============================================================
# TAHMİN ENDPOINT'LERİ
# ============================================================

@app.post("/predict", response_model=PredictResponse, summary="Tek molekül tahmini")
def predict(
    req: PredictRequest,
    enrich: bool = Query(True, description="ChEMBL'den otomatik zenginleştir"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = run_prediction(req.smiles.strip())
    if result is None:
        raise HTTPException(status_code=400, detail="Geçersiz SMILES formatı")

    chembl_id = chembl_name = None
    if enrich:
        mol_data = chembl_lookup_by_smiles(req.smiles.strip())
        if mol_data:
            chembl_id   = mol_data.get("molecule_chembl_id")
            chembl_name = mol_data.get("pref_name")

    pred = Prediction(
        user_id=current_user.id,
        smiles=req.smiles.strip(),
        chembl_id=chembl_id,
        chembl_name=chembl_name,
        **result,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    return PredictResponse(
        smiles=req.smiles,
        saved_id=pred.id,
        chembl_id=chembl_id,
        chembl_name=chembl_name,
        **result,
    )


@app.post("/predict/batch", summary="CSV ile toplu tahmin")
async def predict_batch(
    file: UploadFile = File(...),
    enrich: bool = Query(False, description="ChEMBL zenginleştirme (yavaşlatır)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sadece .csv dosyası yükleyin")

    contents = await file.read()
    reader   = csv.DictReader(io.StringIO(contents.decode("utf-8")))
    fieldnames = reader.fieldnames or []
    smiles_col = next(
        (c for c in fieldnames if "smiles" in c.lower() or "canonical" in c.lower()), None
    )
    if not smiles_col:
        raise HTTPException(
            status_code=400,
            detail=f"'smiles' sütunu bulunamadı. Sütunlar: {fieldnames}",
        )

    rows = list(reader)
    if len(rows) > 50000:
        raise HTTPException(status_code=400, detail="Maksimum 50000 satır")

    results                                  = []
    active_count = inactive_count = error_count = 0

    for row in rows:
        smiles = row.get(smiles_col, "").strip()
        if not smiles:
            results.append({"smiles": smiles, "error": "Boş SMILES"})
            error_count += 1
            continue

        result = run_prediction(smiles)
        if result is None:
            results.append({"smiles": smiles, "error": "Geçersiz SMILES"})
            error_count += 1
            continue

        chembl_id = chembl_name = None
        if enrich:
            mol_data = chembl_lookup_by_smiles(smiles)
            if mol_data:
                chembl_id   = mol_data.get("molecule_chembl_id")
                chembl_name = mol_data.get("pref_name")

        results.append({
            "smiles":      smiles,
            "chembl_id":   chembl_id,
            "chembl_name": chembl_name,
            **result,
        })

        if result["prediction"] == "active":
            active_count += 1
        else:
            inactive_count += 1

        db.add(Prediction(
            user_id=current_user.id,
            smiles=smiles,
            chembl_id=chembl_id,
            chembl_name=chembl_name,
            **result,
        ))

    job = BatchJob(
        user_id=current_user.id,
        filename=file.filename,
        total=len(rows),
        active_count=active_count,
        inactive_count=inactive_count,
        error_count=error_count,
    )
    db.add(job)
    db.commit()

    return {
        "job_id":   job.id,
        "filename": file.filename,
        "total":    len(rows),
        "active":   active_count,
        "inactive": inactive_count,
        "errors":   error_count,
        "results":  results,
    }

# ============================================================
# GEÇMİŞ ENDPOINT'LERİ
# ============================================================

@app.get(
    "/history/predictions",
    response_model=List[PredictionHistory],
    summary="Kullanıcının tahmin geçmişi",
)
def prediction_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(Prediction)
          .filter(Prediction.user_id == current_user.id)
          .order_by(Prediction.created_at.desc())
          .limit(limit)
          .all()
    )


@app.get(
    "/history/batches",
    response_model=List[BatchSummary],
    summary="Kullanıcının batch iş geçmişi",
)
def batch_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(BatchJob)
          .filter(BatchJob.user_id == current_user.id)
          .order_by(BatchJob.created_at.desc())
          .all()
    )

# ============================================================
# ChEMBL ENDPOINT'LERİ
# ============================================================

@app.get("/chembl/molecule", summary="SMILES ile ChEMBL molekül kartı")
def chembl_molecule(
    smiles: str = Query(..., description="Aranacak SMILES"),
    current_user: User = Depends(get_current_user),
):
    mol = chembl_lookup_by_smiles(smiles)
    if not mol:
        return {"found": False, "message": "ChEMBL'de bu molekül bulunamadı"}

    cid     = mol.get("molecule_chembl_id")
    props   = mol.get("molecule_properties") or {}
    structs = mol.get("molecule_structures") or {}
    ro5     = int(props.get("num_ro5_violations") or 0)
    bioacts = chembl_bioactivities(cid, limit=15)

    return {
        "found":           True,
        "chembl_id":       cid,
        "name":            mol.get("pref_name") or "—",
        "max_phase":       mol.get("max_phase"),
        "molecule_type":   mol.get("molecule_type"),
        "formula":         props.get("full_molformula"),
        "mw":              props.get("full_mwt"),
        "alogp":           props.get("alogp"),
        "hba":             props.get("hba"),
        "hbd":             props.get("hbd"),
        "tpsa":            props.get("psa"),
        "rotatable_bonds": props.get("rtb"),
        "ro5_violations":  ro5,
        "ro5_pass":        ro5 == 0,
        "smiles":          structs.get("canonical_smiles"),
        "inchi_key":       structs.get("standard_inchi_key"),
        "bioactivities":   bioacts,
    }


@app.get("/chembl/similar", summary="Benzer moleküller (Tanimoto)")
def chembl_similar(
    smiles: str    = Query(..., description="Referans SMILES"),
    threshold: int = Query(70, ge=40, le=100, description="Tanimoto eşiği (40-100)"),
    limit: int     = Query(10, ge=1,  le=50,  description="Maksimum sonuç"),
    current_user: User = Depends(get_current_user),
):
    results = chembl_similar_molecules(smiles, threshold=threshold, limit=limit)
    return {
        "query_smiles": smiles,
        "threshold":    threshold,
        "count":        len(results),
        "molecules":    results,
    }


@app.get("/chembl/molecule/{chembl_id}", summary="ChEMBL ID ile molekül detayı")
def chembl_by_id(
    chembl_id: str,
    current_user: User = Depends(get_current_user),
):
    detail = chembl_molecule_detail(chembl_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"{chembl_id} bulunamadı")
    bioacts = chembl_bioactivities(chembl_id, limit=15)
    return {**detail, "bioactivities": bioacts}


@app.get("/chembl/targets", summary="Alzheimer ile ilgili ChEMBL hedefleri")
def chembl_alzheimer_targets(current_user: User = Depends(get_current_user)):
    data = _chembl_get("target.json", {
        "target_type":          "SINGLE PROTEIN",
        "organism":             "Homo sapiens",
        "pref_name__icontains": "acetylcholinesterase",
        "limit":                5,
    })
    results = []
    if data:
        for t in data.get("targets", []):
            results.append({
                "chembl_id": t.get("target_chembl_id"),
                "name":      t.get("pref_name"),
                "type":      t.get("target_type"),
                "organism":  t.get("organism"),
            })
    return {"targets": results}


@app.post("/chembl/predict-and-enrich", summary="Tahmin + tam ChEMBL zenginleştirme")
def predict_and_enrich(
    req: PredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Tek endpoint: ML tahmini + ChEMBL tam molekül kartı + benzer moleküller."""
    ml_result = run_prediction(req.smiles.strip())
    if ml_result is None:
        raise HTTPException(status_code=400, detail="Geçersiz SMILES formatı")

    mol           = chembl_lookup_by_smiles(req.smiles.strip())
    chembl_data   = None
    bioactivities = []
    similar       = []

    if mol:
        cid     = mol.get("molecule_chembl_id")
        props   = mol.get("molecule_properties") or {}
        structs = mol.get("molecule_structures") or {}
        ro5     = int(props.get("num_ro5_violations") or 0)
        chembl_data = {
            "chembl_id":       cid,
            "name":            mol.get("pref_name") or "—",
            "max_phase":       mol.get("max_phase"),
            "formula":         props.get("full_molformula"),
            "mw":              props.get("full_mwt"),
            "alogp":           props.get("alogp"),
            "hba":             props.get("hba"),
            "hbd":             props.get("hbd"),
            "tpsa":            props.get("psa"),
            "rotatable_bonds": props.get("rtb"),
            "ro5_violations":  ro5,
            "ro5_pass":        ro5 == 0,
            "smiles":          structs.get("canonical_smiles"),
            "inchi_key":       structs.get("standard_inchi_key"),
            "molecule_type":   mol.get("molecule_type"),
        }
        bioactivities = chembl_bioactivities(cid, limit=10)
        similar       = chembl_similar_molecules(req.smiles.strip(), threshold=70, limit=5)

    pred = Prediction(
        user_id=current_user.id,
        smiles=req.smiles.strip(),
        chembl_id=chembl_data["chembl_id"] if chembl_data else None,
        chembl_name=chembl_data["name"] if chembl_data else None,
        **ml_result,
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)

    return {
        "saved_id":      pred.id,
        "smiles":        req.smiles.strip(),
        "ml":            ml_result,
        "chembl":        chembl_data,
        "bioactivities": bioactivities,
        "similar":       similar,
    }

# ============================================================
# ADMIN ENDPOINT'LERİ
# ============================================================

@app.get("/admin/users", summary="Tüm kullanıcılar (sadece admin)")
def all_users(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Yetkisiz erişim")
    return [
        {
            "id":            u.id,
            "email":         u.email,
            "full_name":     u.full_name,
            "role":          u.role,
            "tahmin_sayisi": len(u.predictions),
            "created_at":    u.created_at,
        }
        for u in db.query(User).all()
    ]


@app.get("/admin/stats", summary="Sistem istatistikleri (sadece admin)")
def system_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Yetkisiz erişim")
    total_users    = db.query(User).count()
    total_preds    = db.query(Prediction).count()
    active_preds   = db.query(Prediction).filter(Prediction.prediction == "active").count()
    total_batches  = db.query(BatchJob).count()
    chembl_matched = db.query(Prediction).filter(Prediction.chembl_id.isnot(None)).count()
    return {
        "toplam_kullanici": total_users,
        "toplam_tahmin":    total_preds,
        "aktif_tahmin":     active_preds,
        "inaktif_tahmin":   total_preds - active_preds,
        "toplam_batch":     total_batches,
        "chembl_eslesen":   chembl_matched,
        "aktif_oran":       round(active_preds / total_preds * 100, 1) if total_preds > 0 else 0,
    }


@app.get("/health", summary="Sistem sağlık kontrolü")
def health():
    return {
        "status":        "ok",
        "model_loaded":  MODEL is not None,
        "model_type":    type(MODEL).__name__ if MODEL else None,
        "threshold":     THRESHOLD,
        "feature_count": len(FEATURE_COLS) if FEATURE_COLS else 0,
        "target_cols":   len(TARGET_COLS),   # ← kaç target sütunu yüklendi
        "chembl_api":    CHEMBL_API,
        "version":       "2.2.0",
    }