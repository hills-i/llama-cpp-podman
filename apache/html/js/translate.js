// translate.js - Translation functionality for the JA <-> EN Translation App
loadSideMenu();

const prompt_ja_to_en = `# Role
Japanese to English translator

# Task
Translate {Input Text} into English.

# Input Text
`;

const prompt_en_to_ja = `# Role
English to Japanese translator

# Task
Translate {Input Text} into Japanese.

# Input Text
`;

const prompt_suffix = `

# Output Format
<translated sentences>
`;

initTranslateUI(prompt_ja_to_en, prompt_en_to_ja, prompt_suffix);
