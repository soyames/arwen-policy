from __future__ import annotations

from typing import List
from ..sources.adapter import GenericSourceAdapter
from .base import BaseRIRAdapter


class ARINAdapter(BaseRIRAdapter):
    """ARIN (American Registry for Internet Numbers) adapter refinement"""

    def __init__(self, source: SourceDefinition):
        super().__init__(source, "ARIN", ["us", "ca"])

    @property
    def adapter_name(self) -> str:
        return "arin-web-adapter"

    def _discover_docs(self, domain: str) -> List[str]:
        """Discover ARIN-specific documents"""
        base_paths = [
            "delegated", "statistics", "whois", "ipv4", "ipv6",
            " allocations", " policyinfo", " hot-certificates"
        ]
        return [
            f"https://www.arin.net/{path}"
            for path in base_paths
        ] + [
            f"https://www.arin.net/statistics/{year}/",
            f"https://www.arin.net/statistics/current/_allocation_v4.csv",
            f"https://www.arin.net/statistics/current/_allocation_v6.csv"
        ]


class RIPENACCAdapter(BaseRIRAdapter):
    """RIPE NCC (RIPE Network Coordination Centre) adapter refinement"""

    def __init__(self, source: SourceDefinition):
        super().__init__(source, "RIPE NCC", ["eu", "uk"])
        self.base_url = "https://www.ripe.net"

    @property
    def adapter_name(self) -> str:
        return "ripe-ncc-web-adapter"

    def _discover_docs(self, domain: str):
        """Discover RIPE NCC-specific documents"""
        key_pages = [
            "db/web/", "statistics", "measurements", "publications",
            "rpki", "certificates", "abuse-mailbox", "ripemdb"
        ]
        return [
            f"{self.base_url}/{page}"
            for page in key_pages
        ] + [
            f"{self.base_url}/statistics/traffic/",
            f"{self.base_url}/statistics/volunteers/",
            f"{self.base_url}/policy/"
        ]


class APNICAdapter(BaseRIRAdapter):
    """APNIC (Asia-Pacific Network Information Centre) adapter refinement"""

    def __init__(self, source: SourceDefinition):
        super().__init__(source, "APNIC", ["au", "asia", "pacific"])
        self.base_url = "https://www.apnic.net"

    @property
    def adapter_name(self) -> str:
        return "apnic-web-adapter"

    def _discover_docs(self, domain: str):
        """Discover APNIC-specific documents"""
        key_pages = [
            "statistics", "whois", "delegations", "publications",
            "abuse", "ipv6-delegations", "reverse-delegations"
        ]
        return [
            f"{self.base_url}/{page}"
            for page in key_pages
        ] + [
            f"{self.base_url}/statistics/",
            f"{self.base_url}/delegations/au/",
            f"{self.base_url}/delegations/jp/"
        ]


class LACNICAdapter(BaseRIRAdapter):
    """LACNIC (Latin Amer and Caribbean Internet Coordination) adapter refinement"""

    def __init__(self, source: SourceDefinition):
        super().__init__(source, "LACNIC", ["la", "lng"])
        self.base_url = "https://www.lacnic.net"

    @property
    def adapter_name(self) -> str:
        return "lacnic-web-adapter"

    def _discover_docs(self, domain: str):
        """Discover LACNIC-specific documents"""
        key_pages = [
            "publications", "statistics", "delegations", "whois",
            "abuse", "policy", "handover-processes"
        ]
        return [
            f"{self.base_url}/{page}"
            for page in key_pages
        ] + [
            f"{self.base_url}/statistics/",
            f"{self.base_url}/delegations/",
            f"{self.base_url}/whois/"
        ]


class AFRINCAdapter(BaseRIRAdapter):
    """AFRINIC (African Internet Registries) adapter refinement"""

    def __init__(self, source: SourceDefinition):
        super().__init__(source, "AFRINC", ["af", "kenya", "nigeria", "na"])
        self.base_url = "https://www.afrinic.net"

    @property
    def adapter_name(self) -> str:
        return "afranic-web-adapter"

    def _discover_docs(self, domain: str):
        """Discover AFRINC-specific documents"""
        key_pages = [
            "publications", "delegations", "statistics", "whois",
            "abuse", "policy", "training"
        ]
        return [
            f"{self.base_url}/{page}"
            for page in key_pages
        ] + [
            f"{self.base_url}/statistics/",
            f"{self.base_url}/delegations/",
            f"{self.base_url}/whois/"
        ]


# Register the adapters to make them discoverable
ADAPTER_CLASSES = [
    ARINAdapter,
    RIPENACCAdapter,
    APNICAdapter,
    LACNICAdapter,
    AFRINCAdapter
]