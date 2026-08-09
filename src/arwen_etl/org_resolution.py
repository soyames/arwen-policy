from __future__ import annotations

from typing import Any, Dict, List, Set, Optional, Tuple
import re
from collections import Counter, defaultdict

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore[assignment]
    _SPACY_AVAILABLE = False

class OrganizationResolver:
    """Resolve organization entities and link to known knowledge base"""

    def __init__(self):
        if _SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")  # type: ignore[union-attr]
            except Exception:
                self.nlp = None
        else:
            self.nlp = None
        self.org_cache: Dict[str, Dict] = {}
        self.known_organizations = self._load_known_orgs()

    def _load_known_orgs(self) -> Dict[str, Dict]:
        """Load known organization database (could be external KB)"""
        # In practice, this would load from a file or database
        return {
            "ICANN": {"type": "internet_governance", "aliases": ["Internet Corporation for Assigned Names and Numbers"]},
            "IGF": {"type": "internet_governance", "aliases": ["Internet Governance Forum"]},
            "IETF": {"type": "technical_standards", "aliases": ["Internet Engineering Task Force"]},
            "ITU": {"type": "intergovernmental", "aliases": ["International Telecommunication Union"]},
            "UN": {"type": "intergovernmental", "aliases": ["United Nations"]},
            "UN DESA": {"type": "intergovernmental", "aliases": ["United Nations Department of Economic and Social Affairs"]},
            "UNESCO": {"type": "intergovernmental", "aliases": ["United Nations Educational, Scientific and Cultural Organization"]},
            "OECD": {"type": "intergovernmental", "aliases": ["Organisation for Economic Co-operation and Development"]},
            "ISOC": {"type": "civil_society", "aliases": ["Internet Society"]},
            "ARIN": {"type": "regional_internet_registry", "aliases": ["American Registry for Internet Numbers"]},
            "RIPE NCC": {"type": "regional_internet_registry", "aliases": ["Réseaux IP Européens Network Coordination Centre"]},
            "APNIC": {"type": "regional_internet_registry", "aliases": ["Asia-Pacific Network Information Centre"]},
            "LACNIC": {"type": "regional_internet_registry", "aliases": ["Latin America and Caribbean Network Information Centre"]},
            "AFRINIC": {"type": "regional_internet_registry", "aliases": ["African Network Information Centre"]},
        }

    def resolve_organization(self, text: str) -> Dict[str, Any]:
        """Resolve organization mentions to canonical entities."""
        if not self.nlp:
            return {}
        doc = self.nlp(text)
        resolved = {}

        for ent in doc.ents:
            if ent.label_ == "ORG":
                canonical = self._find_canonical(ent.text)
                if canonical:
                    resolved[ent.text] = {
                        "canonical": canonical,
                        "type": self.known_organizations[canonical]["type"],
                        "confidence": 0.9
                    }

        return resolved

    def _find_canonical(self, org_text: str) -> Optional[str]:
        """Find canonical name for organization mention"""
        # Direct match
        if org_text in self.known_organizations:
            return org_text

        # Fuzzy match against known aliases
        for canonical, info in self.known_organizations.items():
            for alias in info.get("aliases", []):
                if alias.lower() in org_text.lower() or org_text.lower() in alias.lower():
                    return canonical

        # Abbreviation matching
        if len(org_text.split()) > 1:
            abbrev = "".join(w[0] for w in org_text.split())
            if abbrev in self.known_organizations:
                return abbrev

        return None

    def extract_all_organizations(self, documents: List[str]) -> Dict[str, int]:
        """Extract all organizations from document collection"""
        org_counts = Counter()

        for doc_text in documents:
            resolved = self.resolve_organization(doc_text)
            for org, info in resolved.items():
                canonical = info["canonical"]
                org_counts[canonical] += 1

        return dict(org_counts)

    def organization_network(self, documents: List[str]) -> Dict[str, Dict[str, int]]:
        """Build co-occurrence network of organizations"""
        network = defaultdict(lambda: defaultdict(int))

        for doc_text in documents:
            resolved = self.resolve_organization(doc_text)
            orgs = list(resolved.keys())

            for i, org1 in enumerate(orgs):
                for org2 in orgs[i+1:]:
                    canonical1 = resolved[org1]["canonical"]
                    canonical2 = resolved[org2]["canonical"]
                    if canonical1 != canonical2:
                        network[canonical1][canonical2] += 1
                        network[canonical2][canonical1] += 1

        return {k: dict(v) for k, v in network.items()}

    def link_stakeholders_to_organizations(self, stakeholders: Dict[str, Set[str]], org_resolution: Dict[str, Any]) -> Dict[str, Dict]:
        """Link stakeholder roles to resolved organizations"""
        linked = {}

        for stakeholder, roles in stakeholders.items():
            matched_org = None
            for org_text, info in org_resolution.items():
                # Simple heuristic: stakeholder name contains org name or vice versa
                if (stakeholder.lower() in org_text.lower() or
                    org_text.lower() in stakeholder.lower()):
                    matched_org = info
                    break

            linked[stakeholder] = {
                "roles": roles,
                "organization": matched_org
            }

        return linked

# Public API — instantiate safely; NLP components are optional.
org_resolver = OrganizationResolver()