from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import re
import spacy

# Load English pipeline (used for positional dependency parsing)
nlp = spacy.load("en_core_web_sm")

class PolicyExtractor:
    """Core policy intelligence engine for Phase 4"""

    def __init__(self):
        # Positional argument structure templates
        self.argument_templates = {
            'cause': ('{arg}', '=>', '{result}'),
            'contingency': ('{event}', ',', '{outcome}'),
            'comparison': ('{entity} is', 'more', '{adjective} than {entity2}'),
            'regulation': ('{action}', 'shall', '{compliance_requirement}'),
            'recommendation': ('{actor}', 'should', '{action}'),
        }

    def extract_positions(self, text: str) -> List[Dict[str, str]]:
        """Extract policy positions/assertions from text"""
        doc = nlp(text)
        positions = []

        # Simple heuristic: look for modal verbs and assertions
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(aux in sent_text for aux in ['shall', 'must', 'should', 'may', 'will', 'may']):
                # Basic extraction
                positions.append({
                    'text': sent.text.strip(),
                    'modal': sent._.govmod.root.text if sent._.govmod else None,
                    'confidence': 0.85  # Heuristic confidence
                })

        return positions

    def stakeholder_extraction(self, text: str) -> Dict[str, Set[str]]:
        """Extract stakeholders by entity recognition and relationship parsing"""
        doc = nlp(text)

        stakeholders = defaultdict(set)
        policy_factors = defaultdict(set)

        # Identify organizations and roles
        for ent in doc.ents:
            if ent.label_ in {'ORG', 'PRODUCT', 'WORK_OF_ART'}:
                stakeholders[ent.text].add('organization')

            # Role-based stakeholders
            if ent.label_ in {'PERSON'}:
                roles = self._extract_roles(ent.text)
                for role in roles:
                    stakeholders[role].add('role')

        # Identify policy factors mentioned in text
        for tok in doc:
            if tok.pos_ == "AUX":
                aux_map = {'must': 'requirement', 'should': 'recommendation',
                          'shall': 'requirement', 'can': 'permission', 'may': 'permission'}
                if aux_map.get(tok.text.lower()) and tok.dep_ in {'ROOT', 'REFERENT'}:
                    policy_factors[tok.i] = aux_map[tok.text.lower()]

        return dict(stakeholders), dict(policy_factors)

    def _extract_roles(self, text: str) -> List[str]:
        """Heuristic role extraction from text"""
        role_patterns = [
            r'admin(?:istratrat)?r?e?n?s?i?c?s? ?[A-Z]?\w*',
            r'policy [A-Z]?\w*',
            r'team ?[A-Z]?\w*',
            r'group ?[A-Z]?\w*',
            r'lead(?:ership)? [A-Z]?\w*'
        ]
        for pat in role_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            if matches:
                return matches
        return ['organization']

    def position_extraction(self, text: str) -> List[Dict[str, str]]:
        """Extract policy positions with context"""
        doc = nlp(text)
        positions = []

        for sent in doc.sents:
            # Look for dependency patterns indicative of policy positions
            obj = [tok.text for tok in sent if tok.dep_ == "dobj"]
            subj = [tok.text for tok in sent if tok.dep_ == "nsubj"]

            if obj:
                positions.append({
                    'position': obj[0],
                    'subject': subj[0] if subj else None,
                    'confidence': 0.8  # Heuristic
                })

        return positions

    def argument_extraction(self, text: str) -> List[Dict[str, str]]:
        """Extract argument claims with structure"""
        claims = []
        sent_list = sent_tokenize(text)  # Simple sentence split

        for sent in sent_list:
            sent_doc = nlp(sent)
            # Basic argument pattern: claim + justification
            if any(word in sent.lower() for word in ['because', 'therefore', 'thus', 'hence']):
                claim_parts = sent.split('because')
                if len(claim_parts) == 2:
                    claims.append({
                        'claim': claim_parts[0].strip(),
                        'justification': claim_parts[1].strip()
                    })

        return claims

    def counterargument_detection(self, main_text: str, counter_text: str) -> bool:
        """Heuristic counterargument detection"""
        main_doc = nlp(main_text.lower())
        counter_doc = nlp(counter_text.lower())

        # Look for negation and opposition markers
        negation_patterns = ['however', 'but', 'although', 'despite', 'yet', 'nevertheless']
        opposition_patterns = ['argue', 'claim', 'suggest', 'propose', 'believe']

        has_negation = any(tok.text in negation_patterns for tok in counter_doc)
        has_opposition = any(tok.text in opposition_patterns for tok in main_doc)

        return has_negation or has_opposition

    def evidence_linking(self, claim_text: str, evidence_text: str) -> float:
        """Simple evidence relevance scoring"""
        # Cosine similarity based on TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer().fit([claim_text, evidence_text])
        tfidf_matrix = tfidf.transform([claim_text, evidence_text])

        # Cosine similarity calculation
        from scipy.spatial.distance import cosine
        similarity = 1 - cosine(tfidf_matrix[0], tfidf_matrix[1])
        return float(similarity)

    def temporal_position_tracking(self, text: str) -> List[Dict[str, str]]:
        """Track policy positions along temporal markers"""
        doc = nlp(text)
        temporal_markers = {'before', 'after', 'subsequently', 'later', 'finally', 'initially'}

        temporal_positions = []
        prev_position = None

        for sent in doc.sents:
            if any(tok.text.lower() in temporal_markers for tok in sent):
                current_position = sent.text.strip()
                temporal_positions.append({
                    'position': current_position,
                    'reference_to': prev_position,
                    'temporal_marker': ship
                })
                prev_position = current_position

        return temporal_positions

# Utility function
def sent_tokenize(text: str) -> List[str]:
    """Simple sentence tokenization"""
    import re
    # Split on punctuation followed by space
    return re.split(r'(?<=[.!?])\s+', text)

# Public API
policy_extractor = PolicyExtractor()