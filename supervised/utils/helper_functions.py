"""Helper functions for data preprocessing and categorization."""

def categorize_hr(hr):
    """Categorize heart rate into bradycardic, normal, or tachycardic."""
    if hr < 60:
        return "bradycardic"
    elif hr > 100:
        return "tachycardic"
    return "normal"

def categorize_resp(resp):
    """Categorize respiratory rate into low, normal, or high."""
    if resp < 12:
        return "low"
    elif resp > 20:
        return "high"
    return "normal"

def categorize_pulseox(po):
    """Categorize pulse oximetry into low or normal."""
    return "low" if po < 92 else "normal"

def categorize_bp(bp):
    """Categorize blood pressure into low, normal, or high."""
    if bp < 90:
        return "low"
    elif bp > 140:
        return "high"
    return "normal"

def categorize_temp(temp):
    """Categorize temperature into hypothermic, normal, or febrile."""
    if temp < 97.0:
        return "hypothermic"
    elif temp > 99.5:
        return "febrile"
    return "normal"

def categorize_dbp(dbp):
    """Categorize diastolic blood pressure into low, normal, or high."""
    if dbp < 60:
        return "low"
    elif dbp > 90:
        return "high"
    return "normal"

def categorize_pain(pain):
    """Categorize pain level into no pain, mild, moderate, or severe."""
    if pain == 0:
        return "no pain"
    elif pain <= 3:
        return "mild"
    elif pain <= 6:
        return "moderate"
    return "severe"

def categorize_acuity(acuity):
    """Categorize acuity level."""
    try:
        return f"acuity_{int(acuity)}"
    except Exception:
        return "unknown"

def categorize_age(age):
    """Categorize age into child, young adult, adult, middle aged, or senior."""
    try:
        age = float(age)
    except:
        return "unknown"
    if age < 18:
        return "child"
    elif age < 36:
        return "young_adult"
    elif age < 56:
        return "adult"
    elif age < 76:
        return "middle_aged"
    else:
        return "senior"

def is_daytime(time):
    """Check if the given time is during daytime (7am-7pm)."""
    return 7 <= time.hour < 19

def calculate_sirs(temp, hr, resp):
    """Calculate SIRS (Systemic Inflammatory Response Syndrome) criteria."""
    count = 0
    if (temp > 38) or (temp < 36):
        count += 1
    if hr > 90:
        count += 1
    if resp > 20:
        count += 1
    return 1 if count >= 2 else 0
