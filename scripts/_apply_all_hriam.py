"""Apply ALL HRIAM changes to build_sft_v2.py in one pass."""
original = open('scripts/build_sft_v2.py', encoding='utf-8').read()

# 1) Replace TASK_TYPES block with DOCUMENT + POLICY + HRIAM split
old_block = '# Task types\nTASK_TYPES = [\n    "document_understanding",\n    "evidence_extraction",\n    "policy_question",\n    "stakeholder_position",\n    "argument_identification",\n    "historical_context",\n    "institutional_role",\n    "policy_comparison",\n    "tradeoff_analysis",\n]'
new_block = '''# Task types
# Document-grounded tasks (require a corpus document)
DOCUMENT_TASK_TYPES = [
    "document_understanding",
    "evidence_extraction",
    "policy_question",
    "stakeholder_position",
    "argument_identification",
    "historical_context",
    "institutional_role",
    "policy_comparison",
    "tradeoff_analysis",
]

# Policy-analysis tasks (do NOT require a source document)
POLICY_TASK_TYPES = [
    "multistakeholder_analysis",
    "stakeholder_disagreement",
    "policy_recommendation",
    "perspective_vs_position",
    "uncertainty_handling",
]

# HRIAM tasks - human rights impact assessment & management
HRIAM_TASK_TYPES = [
    "rights_holder_identification",
    "rights_impact_analysis",
    "positive_negative_impacts",
    "disproportionate_impact",
    "stakeholder_rights_mapping",
    "participation_assessment",
    "accountability_remedy",
    "mitigation_safeguards",
    "rights_tradeoff_analysis",
    "lifecycle_hria",
    "panel_analysis",
]

POLICY_TASK_TYPES = POLICY_TASK_TYPES + HRIAM_TASK_TYPES

TASK_TYPES = DOCUMENT_TASK_TYPES + POLICY_TASK_TYPES'''

content = original.replace(old_block, new_block)
assert 'HRIAM_TASK_TYPES' in content, 'Failed: task types'
print('1. Task types OK')

# 2) Update teacher prompt: add HRIAM state guidance before "Return VALID JSON only"
old_skip = 'If the document lacks sufficient evidence for the task, return {"skip": true}.\n- Return VALID JSON only. No markdown, no explanation outside the JSON.'
new_skip = '''If the document has NO policy relevance whatsoever, return {"skip": true}.
  But nearly all documents from policy institutions have some policy relevance -
  do not skip just because the document does not directly answer the question.

HRIAM STATE - for every example, determine whether human-rights analysis is
materially relevant and add an "hriam_state" field:

  "HRIAM_NOT_MATERIAL" - No meaningful human-rights dimension. Answer as a
  normal policy/technical/governance question. Do NOT manufacture a
  human-rights analysis. Example: "What are the technical advantages of IPv6?"

  "HRIAM_RELEVANT" - Human-rights implications exist. Briefly identify
  relevant rights, rights-holders, or potential impacts while maintaining
  the broader multistakeholder policy analysis. Do NOT perform a full HRIA.
  Example: "What are the governance implications of DNS filtering?"

  "HRIAM_CENTRAL" - Human-rights impacts are central. Perform substantive
  HRIAM reasoning: affected rights, rights-holders, stakeholders,
  duty-bearers where applicable, impacts (positive and adverse), trade-offs,
  safeguards, accountability, remedy, and uncertainty.
  Example: "How could DNS filtering affect freedom of expression?"

CRITICAL: Do NOT over-trigger HRIAM. Most policy questions are
HRIAM_NOT_MATERIAL or HRIAM_RELEVANT. Reserve HRIAM_CENTRAL for questions
where human-rights impacts are explicitly the focus.

- Return VALID JSON only. No markdown, no explanation outside the JSON.'''
content = content.replace(old_skip, new_skip)
assert 'HRIAM_NOT_MATERIAL' in content, 'Failed: teacher rules'
print('2. Teacher rules OK')

# 3) Update output format: add hriam_state field
old_fmt = '''  "stakeholders_mentioned": ["from text only"],
  "policy_topics": ["from text only"],
  "uncertainty": "what the document does NOT establish",'''
new_fmt = '''  "stakeholders_mentioned": ["stakeholders relevant to this policy issue"],
  "policy_topics": ["relevant policy domains"],
  "hriam_state": "HRIAM_NOT_MATERIAL | HRIAM_RELEVANT | HRIAM_CENTRAL",
  "uncertainty": "what the document does NOT establish and what remains contested",'''
content = content.replace(old_fmt, new_fmt)
assert 'hriam_state' in content, 'Failed: output format'
print('3. Output format OK')

# 4) Update parse_teacher_response to carry hriam_state
old_ptr = '"stakeholders_mentioned": parsed.get("stakeholders_mentioned", []),\n        "policy_topics": parsed.get("policy_topics", []),'
new_ptr = '"stakeholders_mentioned": parsed.get("stakeholders_mentioned", []),\n        "policy_topics": parsed.get("policy_topics", []),\n        "hriam_state": parsed.get("hriam_state", "HRIAM_RELEVANT"),'
content = content.replace(old_ptr, new_ptr)
assert 'hriam_state' in content[content.find('def parse_teacher_response'):], 'Failed: parse_teacher_response'
print('4. parse_teacher_response OK')
with open('scripts/build_sft_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('ALL DONE (task types + teacher prompt + output format + parse)')
