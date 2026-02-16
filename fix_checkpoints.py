#!/usr/bin/env python3
"""Convert remaining checkpoint() calls to expander format with feedback inside."""

import re


def convert_checkpoint_to_expander(content, key_name):
    """Convert a checkpoint() call to an expander with feedback inside."""

    # Pattern to match checkpoint call and its feedback
    pattern = (
        r'( +)checkpoint\(\s*'
        r'"([^"]+)",\s*'
        r'\[([^\]]+)\],\s*'
        r'key="' + key_name + r'",\s*'
        r'(?:help_text="[^"]*",\s*)?'
        r'\)\s*'
        r'if st\.session_state\.get\("' + key_name + r'"\):\s*'
        r'if st\.session_state\.' + key_name + r' == "([^"]+)":\s*'
        r'st\.success\("([^"]+)"\)\s*'
        r'else:\s*'
        r'st\.info\("([^"]+)"\)'
    )

    replacement = (
        r'\1with st.expander("Checkpoint (1 minute)", expanded=False):\n'
        r'\1    st.markdown("**Question:** \2")\n'
        r'\1    st.radio("Your answer", [\3], key="' + key_name + r'")\n'
        r'\1    if st.session_state.get("' + key_name + r'"):\n'
        r'\1        if st.session_state.' + key_name + r' == "\4":\n'
        r'\1            st.success("\5")\n'
        r'\1        else:\n'
        r'\1            st.info("\6")'
    )

    return re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)


# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Apply conversions for remaining checkpoints
original_len = len(content)
for key in ['cp_data', 'cp_prep', 'cp_finbert', 'cp_compare', 'cp_signal']:
    content = convert_checkpoint_to_expander(content, key)

if len(content) != original_len:
    # Backup original
    with open('app.py.backup', 'w', encoding='utf-8') as f:
        f.write(open('app.py', 'r', encoding='utf-8').read())

    # Write updated content
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ Successfully converted remaining checkpoints to expander format")
    print(f"✓ Backup saved to app.py.backup")
else:
    print("⚠ No changes made - pattern might not match")
