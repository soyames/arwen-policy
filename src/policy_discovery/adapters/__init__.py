"""
Adapter subpackage for policy discovery.

Exposes concrete adapter implementations.
- ICANNAdapter: International Corporation for Assigned Names and Numbers
- IGFAdapter: Internet Governance Forum
- IETFAdapter: Internet Engineering Task Force (if exists)
- ITUTAdapter: International Telecommunication Union
- GovernmentAdapter: National/Regional Government Policies
- AcademicAdapter: Academic & Research Institutions
"""

from .base_adapter import BaseAdapter
from .icann_adapter import ICANNAdapter
from .igf_adapter import IGFAdapter
from .academic_adapter import AcademicAdapter
from .itut_adapter import ITUTAdapter
from .government_adapter import GovernmentAdapter

__all__ = [
    "BaseAdapter",
    "ICANNAdapter",
    "IGFAdapter",
    "AcademicAdapter",
    "ITUTAdapter",
    "GovernmentAdapter",
]