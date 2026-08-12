# Arwen Policy — Human Rights Impact Assessment & Management (HRIAM)

## Definition

HRIAM is Arwen Policy's integrated analytical framework for assessing and reasoning about the human-rights implications of digital and technology policies. It combines impact assessment (HRIA), due diligence principles (HRDD), and a human-rights-based approach (HRBA/PANEL) into a coherent policy-analysis capability.

HRIAM is **policy reasoning**, not legal advice. Arwen does not issue legal determinations, declare human-rights violations, or replace judicial processes.

## Core Concepts

### HRIA — Human Rights Impact Assessment
The systematic process of identifying and assessing actual and potential human-rights impacts of a policy, project, or activity. Focuses on: what rights are affected, how, for whom, and with what severity.

**References:** Danish Institute for Human Rights HRIA Guidance & Toolbox; UNDP Human Rights Impact of AI Assessment Toolkit (2025).

### HRDD — Human Rights Due Diligence
An ongoing process by which an actor identifies, prevents, mitigates, and accounts for how it addresses its human-rights impacts. Broader than a single assessment — encompasses embedding responsibility, ongoing monitoring, tracking, and communication.

**References:** UN Guiding Principles on Business and Human Rights; OHCHR Guidance on Human Rights Due Diligence for Digital Technology Use (2024).

### HRBA / PANEL — Human Rights-Based Approach
A principles-based framework centering human rights in policy design, implementation, and evaluation. PANEL provides a structured analytical lens:

- **P — Participation:** Meaningful engagement of affected persons
- **A — Accountability:** Responsibility of duty-bearers; monitoring and answerability
- **N — Non-discrimination and Equality:** Differential and disproportionate effects
- **E — Empowerment:** Enabling rights-holders to understand and claim their rights
- **L — Legality:** Grounding in domestic and international human rights law

**References:** Scottish Human Rights Commission; ENNHRI; OHCHR Human Rights-Based Approach.

## Three Analytical Categories

Arwen distinguishes three categories relevant to rights analysis:

### Stakeholders
Actors with institutional interests, influence, expertise, or involvement in a policy process. Defined in `configs/stakeholders.yaml`. Examples: governments, regulators, industry, technical community, civil society organizations.

### Rights-holders
Persons or groups holding internationally recognized human rights that may be affected by a policy. Defined in `configs/stakeholders.yaml`. Examples: users, journalists, children, minorities, persons with disabilities, workers.

### Duty-bearers
Actors with human-rights obligations or responsibilities. States and public authorities are primary duty-bearers under international law. Businesses have a responsibility to respect human rights under the UNGPs — distinct from state obligations. Defined in `configs/stakeholders.yaml`.

## HRIAM Trigger Model

HRIAM is NOT automatically applied. Three explicit states:

| State | Description | Example |
|-------|-------------|---------|
| **HRIAM_NOT_MATERIAL** | No meaningful human-rights dimension | "What is DNSSEC?" |
| **HRIAM_RELEVANT** | Human-rights implications to identify and briefly analyze | "Governance implications of DNS filtering" |
| **HRIAM_CENTRAL** | Human-rights impacts are central to the question | "How could DNS filtering affect freedom of expression?" |

Depth is proportional. HRIAM_NOT_MATERIAL → no HRIAM section. HRIAM_RELEVANT → brief rights-aware analysis. HRIAM_CENTRAL → comprehensive reasoning.

## Impact Model

### Impact ≠ Violation

| Term | Definition | Evidence Required? |
|------|-----------|-------------------|
| Potential impact | Plausible effect requiring assessment | No — analytical reasoning |
| Adverse impact | Potentially negative effect on a right | No — analytical reasoning |
| Actual impact | Evidenced effect that occurred | Yes — source required |
| Violation | Legal/normative conclusion of breach | Yes — requires authority, jurisdiction, evidence |
| Risk | Possibility of future adverse impact | No — analytical reasoning |
| Residual risk | Remaining risk after mitigation | No — analytical reasoning |

### Severity Assessment

Qualitative categories grounded in specific dimensions: magnitude, scope, duration, reversibility, vulnerability, and remedy availability. Categories: Low / Medium / High / Critical.

Severity is NOT determined by the right alone. "Privacy issue = High" is incorrect. Severity depends on the specific mechanism, context, and affected population.

### Positive AND Adverse Impacts

HRIAM analyzes both positive and adverse impacts. It is NOT a "find harms" system. A cybersecurity policy may both protect users from crime AND create privacy risks. Both dimensions must be analyzed.

## Key Distinctions

### Security is NOT a standalone human right
The right to security of person is a recognized human right. Cybersecurity is a legitimate policy objective that supports rights enjoyment. It is not a standalone human right equivalent to freedom of expression or privacy.

### Stakeholder ≠ Rights-holder ≠ Duty-bearer
These are distinct analytical categories. A regulator is a stakeholder, not necessarily a rights-holder. A user may be both. A state is a duty-bearer, not merely a stakeholder.

### Participation ≠ Consultation
Meaningful participation requires accessibility, information, timing, capacity, representation, influence, and the ability to challenge outcomes. A formal consultation does not automatically constitute meaningful participation.

### Remedy ≠ Appeal
Appeal provides access to remedy. Remedy includes correction, restoration, compensation, restitution, rehabilitation, acknowledgment, and guarantees against recurrence.

## Necessity, Proportionality, and Alternatives

When a policy restricts rights, Arwen considers:
1. Legitimate objective
2. Suitability / rational connection
3. Necessity (are less restrictive alternatives available?)
4. Proportionality (does the benefit justify the interference?)
5. Safeguards
6. Remedy
7. Review

## Lifecycle

HRIAM covers: Design → Policy Adoption → Implementation → Deployment → Monitoring → Emerging Evidence → Reassessment → Policy Adjustment → Decommissioning (where applicable).

## Limitations

1. HRIAM is policy analysis, not legal advice
2. Arwen does not declare human-rights violations
3. Arwen does not replace judicial or oversight processes
4. Severity assessments are qualitative and context-dependent
5. Source-free analysis is general policy reasoning — specific claims require evidence
6. HRIAM is triggered by material relevance, not applied mechanically

## External Methodology References

- Danish Institute for Human Rights, "Human Rights Impact Assessment Guidance and Toolbox" (2016, updated); "HRIA of Digital Activities" (2020)
- OHCHR, "UN Secretary-General's Guidance on Human Rights Due Diligence for Digital Technology Use" (2024)
- UNDP, "Human Rights Impact of AI Assessment Toolkit" (2025)
- OHCHR B-Tech Project — Applying UNGPs to the technology sector
- Scottish Human Rights Commission / ENNHRI — PANEL framework
- UNESCO & OHCHR, "Protecting Critical Voices" (2025)
- Global Partners Digital — Human rights in Internet governance advocacy

## Implementation

- Canonical prompt: `src/arwen_etl/engine/arwen_prompt.py`
- Stakeholder/rights-holder/duty-bearer config: `configs/stakeholders.yaml`
- HRIAM task types and templates: `scripts/build_sft_v2.py`
- Dataset validation: `scripts/validate_dataset.py`
- Behavioral tests: `tests/test_evaluation_suite.py`
- Synthesis pipeline: `src/arwen_engine/pipeline.py`
