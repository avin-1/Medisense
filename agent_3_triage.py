import csv

class TriageAgent:
    def __init__(self, severity_path='MasterData/symptom_severity.csv'):
        self.severity_dict = {}
        self._load_severities(severity_path)

    def _load_severities(self, filepath):
        try:
            with open(filepath, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        # row[0] is symptom name, row[1] is severity
                        self.severity_dict[row[0].strip()] = int(row[1].strip())
        except Exception as e:
            print(f"Warning: Could not load severity dict perfectly: {e}")

    def evaluate_risk(self, extracted_symptoms: list[str]) -> tuple[bool, int]:
        """
        Calculates a real-time risk score. 
        Returns (is_emergency, total_risk_score)
        """
        total_risk = 0
        for sysmptom in extracted_symptoms:
            severity = self.severity_dict.get(sysmptom, 1) # Default to 1 if unknown
            total_risk += severity
            
        # The threshold of 13 is arbitrary based on logic. Let's say if total_risk > 13 it's an emergency.
        # Alternatively, if any SINGLE symptom is severity >= 7 (like high_fever, coma), emergency.
        # Let's combine both logic for a robust triage.
        is_emergency = False
        
        max_single_severity = max([self.severity_dict.get(s, 1) for s in extracted_symptoms] + [0])
        
        if total_risk >= 13 or max_single_severity >= 6:
            is_emergency = True
            
        return is_emergency, total_risk
