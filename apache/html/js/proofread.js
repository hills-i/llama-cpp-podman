// Proofread functionality for the English Email Checker
loadSideMenu();

const prompt_template_1 = `
# Role
English email proofreader

# Task
Review {Mail draft} to ensure the following:

1. Correctness: Check for grammar, punctuation, and spelling errors.
2. Naturalness: Ensure the language flows smoothly and feels natural to native speaker.
3. Politeness: Confirm that the tone is polite, respectful and appropriate for professional correspondence.
4. Clarity: Check the clarity of the email.

Make revisions as needed, but retain the intended meaning and context.

# Context
A business mail.
The email will be sent to a non-English native. Use clear and simple language that is easily understandable.

# Mail draft
\`\`\`
`;

const prompt_template_2 = `
\`\`\`
# Output format
The results of evaluating the original email  about tolerance for acceptable business email on a scale of 1 to 10 for the followings. The passing mark is 6.

Corectness score:
Naturalness score:
Politeness score:
Clarity score:

Whether the original email is acceptable to send as business email (Answer yes or no) :
Proofread email (if there are any) :
Changes (if there are any) :
`;

initProofreadUI(prompt_template_1, prompt_template_2);