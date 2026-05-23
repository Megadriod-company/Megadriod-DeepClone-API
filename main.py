import os
import re
import math
import datetime
import io
from typing import List, Optional, Dict
from fastapi import FastAPI, Request, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# Ensure these are in requirements.txt: librosa, numpy, soundfile, python-multipart
import numpy as np
import librosa
import soundfile as sf

# ==========================================
# 1. DATABASE CONFIGURATION (RENDER & LOCAL)
# ==========================================

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./megadriod_deepclone.db")
if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 2. SQLALCHEMY DATABASE MODELS
# ==========================================

class AuditSession(Base):
    """Tracks a complete DeepClone audit session for a user."""
    __tablename__ = "audit_sessions"
    id = Column(String, primary_key=True, index=True) # UUID
    ip_address = Column(String)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    audio_audits = relationship("AudioAnalysisAudit", back_populates="session")
    transcript_audits = relationship("TranscriptAudit", back_populates="session")
    phone_audits = relationship("PhoneCheckAudit", back_populates="session")

class AudioAnalysisAudit(Base):
    """Stores telemetry for EchoClone, VoiceTwin, and AudioTrap."""
    __tablename__ = "audio_analysis_audits"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("audit_sessions.id"))
    analysis_type = Column(String) # 'EchoClone', 'VoiceTwin', 'AudioTrap'
    file_duration_sec = Column(Float)
    splice_anomalies_detected = Column(Integer)
    ai_confidence_score = Column(Float) # Score indicating likelihood of AI generation
    similarity_percentage = Column(Float, nullable=True) # For VoiceTwin
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("AuditSession", back_populates="audio_audits")

class TranscriptAudit(Base):
    """Stores telemetry for PanicScript."""
    __tablename__ = "transcript_audits"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("audit_sessions.id"))
    transcript = Column(Text)
    manipulation_score = Column(Float)
    detected_tactics = Column(String) # JSON string of tactics (urgency, financial)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("AuditSession", back_populates="transcript_audits")

class PhoneCheckAudit(Base):
    """Stores telemetry for FakeCall Check."""
    __tablename__ = "phone_check_audits"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("audit_sessions.id"))
    phone_number = Column(String)
    risk_level = Column(String) # Safe, Suspicious, High-Risk
    flags = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("AuditSession", back_populates="phone_audits")

Base.metadata.create_all(bind=engine)

# ==========================================
# 3. PYDANTIC SCHEMAS (API REQUESTS/RESPONSES)
# ==========================================

class SessionCreate(BaseModel):
    session_id: str = Field(..., description="Unique UUID for the training session")

class PanicScriptRequest(BaseModel):
    session_id: str
    transcript: str

class PanicScriptResponse(BaseModel):
    manipulation_score_percentage: float
    urgency_flags: List[str]
    financial_flags: List[str]
    fear_flags: List[str]
    verdict: str

class PhoneCheckResponse(BaseModel):
    phone_number: str
    risk_level: str
    is_voip_suspect: bool
    warnings: List[str]

# ==========================================
# 4. FASTAPI APP & DEPENDENCIES
# ==========================================

app = FastAPI(
    title="Megadriod DeepClone API",
    description="Next-Gen Audio Forensics and AI Scam Detection Backend",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 4.5 FRONTEND SERVING ROUTES
# ==========================================

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serves the main index.html file."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found in the main directory."}

@app.get("/logo.png", include_in_schema=False)
def serve_logo():
    """Serves the logo image."""
    if os.path.exists("logo.png"):
        return FileResponse("logo.png")
    return {"error": "logo.png not found in the main directory."}


# ==========================================
# 5. CORE ANALYTICS ENGINES (REAL ALGORITHMS)
# ==========================================

# PanicScript Engine (Real Text Analysis)
MANIPULATION_LEXICON = {
    "urgency": [
        # --- Original (8) ---
        "urgent", "immediately", "asap", "now", "hurry", "quick", "don't hang up", "fast",
        # --- Expanded Urgency (32 Tokens) ---
        "final warning", "expires", "deadline", "running out", "last chance", "instant", 
        "terminate", "critical", "straight away", "freeze", "seconds", "minutes", "hours",
        "clock is ticking", "dont wait", "act now", "without delay", "swiftly", "suspended",
        "temporary", "overnight", "imminent", "impending", "rushed", "sudden", "compulsory",
        "mandatory", "non-negotiable", "breaking", "flash", "unconditional", "last-minute",
        "high priority", "vital", "at once", "today", "tonight", "instantaneous", "quickest", "shortly"
    ],
    
    "financial": [
        # --- Original (10) ---
        "wire", "transfer", "bank", "crypto", "gift card", "bitcoin", "money", "funds", "account", "zelle",
        # --- Expanded Financial (40 Tokens) ---
        "deposit", "payment", "routing number", "cashapp", "venmo", "usdt", "wallet", "overdue", 
        "unpaid", "balance", "invoice", "refund", "card details", "pin", "cvv", "remittance",
        "bvn", "nin", "token", "otp", "kuda", "opay", "palmpay", "moniepoint", "western union",
        "moneygram", "ledger", "ethereum", "solana", "assets", "inheritance", "collateral",
        "transaction", "fee", "penalty", "fine", "billing", "credit", "debit", "withdrawal",
        "currency", "cash", "check", "cheque", "reimbursement", "bribe", "payout", "wages", "salary", "loan"
    ],
    
    "fear": [
        # --- Original (9) ---
        "arrested", "police", "jail", "accident", "hospital", "hacked", "stolen", "warrant", "breach",
        # --- Expanded Fear (41 Tokens) ---
        "lawsuit", "court", "prosecution", "illegal", "seizure", "compromised", "infected", "exposed", 
        "blackmail", "consequences", "shut down", "threat", "leaked", "scandal", "fraudulent", "prison",
        "cell", "handcuffs", "convicted", "guilty", "crime", "felony", "subpoena", "summons", 
        "deportation", "visa cancellation", "kidnapped", "ransom", "danger", "harm", "ruined", 
        "shame", "disgrace", "leak", "videos", "photos", "chat history", "evidence", "defamation", 
        "libel", "sued", "punishment", "penalty", "forfeiture", "eviction", "foreclosure", "blacklisted", "banned"
    ],
    
    "authority": [
        # --- Original (9) ---
        "manager", "director", "officer", "inspector", "government", "irs", "fbi", "cbn", "efcc",
        # --- Expanded Authority (37 Tokens) ---
        "agent", "attorney", "legal department", "fraud division", "headquarters", "investigator", 
        "marshal", "sheriff", "customs", "magistrate", "judge", "prosecutor", "icpc", "dss", 
        "interpol", "embassy", "consulate", "immigration", "civil service", "tax collector", 
        "revenue", "firs", "auditor", "superintendent", "commissioner", "detective", "chief", 
        "governor", "chairman", "board", "admin", "moderator", "support desk", "helpdesk", 
        "security team", "verified personnel", "compliance officer", "ombudsman", "regulatory body", "secret service"
    ]
}

def analyze_panic_script(text: str) -> dict:
    """Uses weighted word-frequency analysis to detect social engineering."""
    text_lower = text.lower()
    results = {"urgency": [], "financial": [], "fear": [], "authority": [], "score": 0.0}
    
    words_found = 0
    total_words = len(text.split())
    if total_words == 0: return results

    for category, keywords in MANIPULATION_LEXICON.items():
        for word in keywords:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                results[category].append(word)
                words_found += 1
                
    # Weighting: Fear + Financial combined is highly indicative of a scam
    base_score = (words_found / max(10, total_words)) * 200 # Normalized baseline
    
    if len(results["financial"]) > 0 and len(results["fear"]) > 0:
        base_score += 40.0 # Huge penalty for combo
    if len(results["urgency"]) > 0:
        base_score += 15.0
        
    results["score"] = min(100.0, base_score)
    return results

# Audio Engine (Real DSP using Librosa)
def process_audio_features(file_bytes: bytes) -> dict:
    """
    Extracts actual mathematical features from the audio for EchoClone & AudioTrap.
    - MFCCs (Mel-frequency cepstral coefficients) map the vocal tract.
    - RMSE (Root Mean Square Energy) detects sudden silences (splices).
    """
    try:
        # Load audio from bytes
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
        
        # FIX: Cast NumPy float to standard Python float
        duration = float(librosa.get_duration(y=y, sr=sr))
        
        # 1. MFCC for Voice Tonal Pattern (EchoClone)
        # Taking the mean of the MFCCs over time gives a vocal "fingerprint"
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1).tolist()
        
        # 2. Splice Detection (AudioTrap)
        # Calculate energy (volume) over time. Sudden, unnatural drops to absolute zero indicate digital splicing.
        rmse = librosa.feature.rms(y=y)[0]
        # Find frames where energy drops dramatically below the mean
        energy_drops = np.where(rmse < (np.mean(rmse) * 0.1))[0]
        
        # Group consecutive drops to count "silence/cut events"
        splices = 0
        if len(energy_drops) > 0:
            diffs = np.diff(energy_drops)
            # FIX: Cast NumPy int to standard Python int
            splices = int(len(np.where(diffs > 5)[0]) + 1) # 5 frames gap indicates a distinct cut

        # 3. Spectral Flatness (AI Generation indicator)
        # AI voices often have a very consistent, "flat" spectral footprint compared to human breath variability.
        
        # FIX: Cast NumPy floats to standard Python floats
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        ai_prob = float(min(100.0, (flatness * 1000))) # Simplified heuristic for demo purposes

        return {
            "success": True,
            "duration": duration,
            "mfcc_fingerprint": mfcc_mean,
            "splices_detected": splices,
            "ai_probability": ai_prob
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def calculate_cosine_similarity(vec1: list, vec2: list) -> float:
    """Real math to compare two voice fingerprints (VoiceTwin)."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    similarity = dot_product / (norm_v1 * norm_v2)
    return float(max(0.0, min(100.0, similarity * 100)))

# ==========================================
# 6. API ENDPOINTS
# ==========================================

@app.post("/api/v1/session/start", status_code=status.HTTP_201_CREATED)
def start_audit_session(data: SessionCreate, request: Request, db: Session = Depends(get_db)):
    """Initializes a new training telemetry session."""
    client_ip = request.client.host if request.client else "unknown"
    db_session = AuditSession(id=data.session_id, ip_address=client_ip)
    db.add(db_session)
    db.commit()
    return {"message": "DeepClone audit session initialized", "session_id": data.session_id}


@app.post("/api/v1/clone/panic-script", response_model=PanicScriptResponse)
def evaluate_panic_script(data: PanicScriptRequest, db: Session = Depends(get_db)):
    """Analyzes text for emotional manipulation and scam tactics."""
    analysis = analyze_panic_script(data.transcript)
    
    score = analysis["score"]
    if score > 70: verdict = "CRITICAL: Highly Manipulative / Likely Scam"
    elif score > 30: verdict = "WARNING: Suspicious Tones Detected"
    else: verdict = "SAFE: Normal Conversation Pattern"

    audit = TranscriptAudit(
        session_id=data.session_id,
        transcript=data.transcript,
        manipulation_score=score,
        detected_tactics=str({k: analysis[k] for k in ["urgency", "financial", "fear"]})
    )
    db.add(audit)
    db.commit()

    return PanicScriptResponse(
        manipulation_score_percentage=round(score, 2),
        urgency_flags=analysis["urgency"],
        financial_flags=analysis["financial"],
        fear_flags=analysis["fear"],
        verdict=verdict
    )


@app.get("/api/v1/clone/fake-call-check/{phone_number}", response_model=PhoneCheckResponse)
def evaluate_phone_number(phone_number: str, session_id: str, db: Session = Depends(get_db)):
    """Algorithmic structural check for scam/VoIP numbers."""
    cleaned = re.sub(r'\D', '', phone_number)
    warnings = []
    risk = "LOW"
    is_voip = False

    # Real Structural Checks
    if not cleaned:
        warnings.append("Invalid number format.")
        risk = "HIGH"
    else:
        # Check for common VoIP/Virtual routing lengths (often hide behind non-standard lengths)
        if len(cleaned) < 10 or len(cleaned) > 15:
            warnings.append("Irregular length - possible spoofed routing.")
            risk = "MEDIUM"
        
        # International Toll Fraud Area Codes (e.g., +1-809, +1-876, +1-284)
        toll_fraud_codes = ["1809", "1876", "1284", "1473"]
        if any(cleaned.startswith(code) for code in toll_fraud_codes):
            warnings.append("Number originates from known One-Ring Scam region.")
            risk = "CRITICAL"
            is_voip = True
            
        # Unallocated / Premium local prefixes (General rule checking)
        if len(cleaned) >= 11 and cleaned.startswith("234"): # Nigeria Example
            prefix = cleaned[3:6]
            if prefix in ["090", "091", "070", "080", "081"]: # Standard
                pass
            else:
                warnings.append("Non-standard carrier prefix. Possible virtual number.")
                is_voip = True
                risk = "MEDIUM"

    audit = PhoneCheckAudit(
        session_id=session_id,
        phone_number=phone_number,
        risk_level=risk,
        flags=str(warnings)
    )
    db.add(audit)
    db.commit()

    return PhoneCheckResponse(
        phone_number=phone_number,
        risk_level=risk,
        is_voip_suspect=is_voip,
        warnings=warnings
    )


@app.post("/api/v1/clone/audio-forensics")
async def process_audio_forensics(
    session_id: str = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Handles EchoClone and AudioTrap.
    Receives a real audio file, mathematically extracts tonal fingerprints, 
    and checks for digital splicing.
    """
    file_bytes = await audio_file.read()
    features = process_audio_features(file_bytes)
    
    if not features["success"]:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {features.get('error')}")

    splices = features["splices_detected"]
    ai_prob = features["ai_probability"]
    
    audit = AudioAnalysisAudit(
        session_id=session_id,
        analysis_type="EchoClone_AudioTrap",
        file_duration_sec=features["duration"],
        splice_anomalies_detected=splices,
        ai_confidence_score=ai_prob
    )
    db.add(audit)
    db.commit()

    return {
        "duration_seconds": round(features["duration"], 2),
        "echo_clone_analysis": {
            "vocal_tract_nodes_mapped": len(features["mfcc_fingerprint"]),
            "ai_generation_probability": round(ai_prob, 2),
            "verdict": "High AI Signature Detected" if ai_prob > 60 else "Human Voice Signature"
        },
        "audio_trap_analysis": {
            "splices_detected": splices,
            "manipulation_status": "MANIPULATED: Audio cuts detected" if splices > 0 else "CLEAN: Continuous audio stream"
        },
        "raw_fingerprint_vector": features["mfcc_fingerprint"] # Sent to client for VoiceTwin comparison
    }


@app.post("/api/v1/clone/voice-twin")
def compare_voice_twins(
    session_id: str = Form(...),
    vector_a: str = Form(..., description="Comma separated MFCC vector for Voice 1"),
    vector_b: str = Form(..., description="Comma separated MFCC vector for Voice 2"),
    db: Session = Depends(get_db)
):
    """
    Takes two vocal fingerprints (generated by EchoClone endpoint) 
    and calculates actual mathematical cosine similarity.
    """
    try:
        vec1 = [float(x) for x in vector_a.split(",")]
        vec2 = [float(x) for x in vector_b.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Vectors must be comma-separated floats.")

    similarity = calculate_cosine_similarity(vec1, vec2)
    
    audit = AudioAnalysisAudit(
        session_id=session_id,
        analysis_type="VoiceTwin",
        file_duration_sec=0.0,
        splice_anomalies_detected=0,
        ai_confidence_score=0.0,
        similarity_percentage=similarity
    )
    db.add(audit)
    db.commit()

    return {
        "similarity_percentage": round(similarity, 2),
        "verdict": "Match (Same Speaker)" if similarity > 85.0 else "Mismatch (Different Speakers)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)