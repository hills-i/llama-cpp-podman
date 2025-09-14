// Reply functionality for the English Email Reply Draft Creation
loadSideMenu();

const prompt_template_1 = `
You are an AI that drafts polite business email replies in English.
Read the "Input Email" below and write an appropriate reply that:

- Is written in natural and professional English suitable for business communication.
- Clearly and politely addresses any questions or requests from the sender.
- Always includes a thank you to the sender for their message or consideration.
- Ends with a courteous closing remark, such as "Best regards" or "Thank you for your continued support."
- **Do not** include a signature (name, company, etc.); write only the email body.

---

### Input Email
\`\`\`
`;

const prompt_template_2 = `
\`\`\`
---
### What to Reply
\`\`\`
`;

const prompt_template_3 = `
\`\`\`
---
### Output Format

**Reply Draft:**

\`\`\`

(Paste the drafted reply here)

\`\`\`
\`\`\`

---

## Example

**Input Email:**

\`\`\`
Hello, Mr. Kevin,

Thank you for your prompt response regarding the matter. I have reviewed the file you sent and found no issues, so please consider it confirmed on our side.
Additionally, could you please update the schedule to reflect today's date as the file submission date?

Thank you for your cooperation.

Best regards,
\`\`\`

**Reply Draft:**

\`\`\`
Hello,

Thank you for reviewing the file and confirming that there are no issues.

I am glad to hear that everything is in order. I will update the schedule to reflect today as the submission date and will send you the revised document shortly.

Thank you again for your continued cooperation.

Best regards,
\`\`\`
`;

initReplyLocalUI(prompt_template_1, prompt_template_2, prompt_template_3);