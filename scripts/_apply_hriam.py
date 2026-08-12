"""Apply all HRIAM changes to build_sft_v2.py cleanly."""

content = open('scripts/build_sft_v2.py', encoding='utf-8').read()

# 1. Add HRIAM task types
old_tt = '# All task types (used for backward compatibility references)\nTASK_TYPES = DOCUMENT_TASK_TYPES + POLICY_TASK_TYPES'
new_tt = '''# HRIAM tasks — human rights impact assessment & management
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

# All policy tasks (including HRIAM)
POLICY_TASK_TYPES = POLICY_TASK_TYPES + HRIAM_TASK_TYPES

# All task types (used for backward compatibility references)
TASK_TYPES = DOCUMENT_TASK_TYPES + POLICY_TASK_TYPES'''
content = content.replace(old_tt, new_tt)
print('1. HRIAM task types added')

# 2. Add hriam_state to output format in teacher prompt
old_of = '''  "stakeholders_mentioned": ["stakeholders relevant to this policy issue"],
  "policy_topics": ["relevant policy domains"],
  "uncertainty": "what the document does NOT establish and what remains contested",'''
new_of = '''  "stakeholders_mentioned": ["stakeholders relevant to this policy issue"],
  "policy_topics": ["relevant policy domains"],
  "hriam_state": "HRIAM_NOT_MATERIAL | HRIAM_RELEVANT | HRIAM_CENTRAL",
  "uncertainty": "what the document does NOT establish and what remains contested",'''
content = content.replace(old_of, new_of)
print('2. hriam_state in output format')

# 3. Add HRIAM state guidance to teacher rules
old_skip = '- If the document has NO policy relevance whatsoever, return {"skip": true}.\n  But nearly all documents from policy institutions have some policy relevance —\n  do not skip just because the document doesn\'t directly answer the question.\n- Return VALID JSON only. No markdown, no explanation outside the JSON.'
new_skip = '''- If the document has NO policy relevance whatsoever, return {"skip": true}.
  But nearly all documents from policy institutions have some policy relevance —
  do not skip just because the document doesn't directly answer the question.

HRIAM STATE — for every example, determine whether human-rights analysis is
materially relevant and add an "hriam_state" field:

  "hriam_state": "HRIAM_NOT_MATERIAL"
    — No meaningful human-rights dimension. Answer as a normal policy,
      technical, or governance question. Do NOT manufacture a human-rights
      analysis. Example: "What are the technical advantages of IPv6?"

  "hriam_state": "HRIAM_RELEVANT"
    — Human-rights implications exist. Briefly identify relevant rights,
      rights-holders, or potential impacts while maintaining the broader
      multistakeholder policy analysis. Do NOT perform a full HRIA.
      Example: "What are the governance implications of DNS filtering?"

  "hriam_state": "HRIAM_CENTRAL"
    — Human-rights impacts are central. Perform substantive HRIAM reasoning:
      affected rights, rights-holders, stakeholders, duty-bearers where
      applicable, impacts (positive and adverse), trade-offs, safeguards,
      accountability, remedy, and uncertainty.
      Example: "How could mandatory DNS filtering affect freedom of expression?"

CRITICAL: Do NOT over-trigger HRIAM. Most policy questions are
HRIAM_NOT_MATERIAL or HRIAM_RELEVANT. Reserve HRIAM_CENTRAL for questions
where human-rights impacts are explicitly the focus.

- Return VALID JSON only. No markdown, no explanation outside the JSON.'''
content = content.replace(old_skip, new_skip)
print('3. HRIAM state guidance in teacher rules')

# 4. Update build_policy_example
old_bpe = '''        "stakeholders_mentioned": template.get("stakeholders_mentioned", []),
        "policy_topics": template.get("policy_topics", []),'''
new_bpe = '''        "stakeholders_mentioned": template.get("stakeholders_mentioned", []),
        "policy_topics": template.get("policy_topics", []),
        "hriam_state": template.get("hriam_state", "HRIAM_RELEVANT"),'''
content = content.replace(old_bpe, new_bpe)
print('4. build_policy_example updated')

# 5. Update parse_teacher_response
old_ptr = '''        "stakeholders_mentioned": parsed.get("stakeholders_mentioned", []),
        "policy_topics": parsed.get("policy_topics", []),'''
new_ptr = '''        "stakeholders_mentioned": parsed.get("stakeholders_mentioned", []),
        "policy_topics": parsed.get("policy_topics", []),
        "hriam_state": parsed.get("hriam_state", "HRIAM_RELEVANT"),'''
content = content.replace(old_ptr, new_ptr)
print('5. parse_teacher_response updated')

# 6. Save (templates will be added separately)
with open('scripts/build_sft_v2.py', 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Done. hriam_state fields: {content.count("hriam_state")}')
print('Templates NOT added — will be done separately')
