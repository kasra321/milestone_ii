import re
import numpy as np
import pandas as pd

def clean_pain_value(value):
    """
    Cleans a raw pain severity value and converts it to a numeric pain score (0-10).
    Handles expressions like '5 out of 10' or '5/10' by extracting the numerator if the denominator is 10.
    If the string cannot be parsed into a number, returns -1 to denote an 'unclear' entry.
    """
    # If the value is already missing, return np.nan.
    if pd.isnull(value):
        return np.nan

    # Convert to string, lowercase, and remove extra quotes and spaces.
    s = str(value).strip().lower()
    s = s.replace('"', '').replace("'", "").strip()
    
    # Define indicators that suggest the value is missing or ambiguous.
    missing_indicators = [
        'unable', 'refus', 'n/a', 'na', 'error', 'not able',
        'non-verbal', 'intubated', 'sleep', 'asleep',
        'uta', 'ett', 'ua', 'unabl', 'unab', 'unale', 'uncooperative', 'no answer'
    ]
    if any(ind in s for ind in missing_indicators):
        return np.nan

    # Look for patterns like "x out of y" or "x/y"
    fraction_pattern = re.search(r'(\d+\.?\d*)\s*(?:out of|\/)\s*(\d+\.?\d*)', s)
    if fraction_pattern:
        num, denom = fraction_pattern.groups()
        try:
            num = float(num)
            denom = float(denom)
            # If denominator is 10, assume the numerator is the pain score.
            if np.isclose(denom, 10):
                score = num
            else:
                # Otherwise, average the two numbers as a fallback.
                score = (num + denom) / 2
            # Cap the value to the typical pain scale (0-10)
            score = max(0, min(score, 10))
            return score
        except Exception:
            pass  # Fall back to further processing

    # Map common qualitative descriptors to numeric pain scores.
    qualitative_mapping = {
        'no pain': 0,
        'none': 0,
        'denies': 0,
        'mild': 2,
        'a little': 2,
        'little': 2,
        'moderate': 5,
        'some': 5,
        'severe': 8,
        'critical': 10,
        'crit': 10,
        'terrible': 10,
        'awful': 10,
        'excruciating': 10,
        'intense': 10,
        'very painful': 10,
        'very bad': 10,
        'a lot': 10,
        'alot': 10
    }
    for key, val in qualitative_mapping.items():
        if key in s:
            return val

    # Use regex to extract any numeric values (handles decimals and numbers in ranges).
    numbers = re.findall(r'\d+\.?\d*', s)
    if numbers:
        try:
            if len(numbers) == 1:
                num = float(numbers[0])
            else:
                # For ranges like "2-3" or "5 to 10", average the numbers.
                num_list = [float(n) for n in numbers]
                num = sum(num_list) / len(num_list)
            # Cap the value to the typical pain scale (0-10)
            num = max(0, min(num, 10))
            return num
        except Exception:
            return -1  # Fallback: unclear

    # If no method worked, return -1 to indicate the value is unclear.
    return -1
