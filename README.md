# 🌐 AI Multilingual FAQ Builder

A business uploads its information. The app writes editable FAQs in **English and Urdu** using only that material.

## Features
- **Your material:** upload PDF, Word, TXT, Markdown or CSV files (up to 5), and/or paste text.
- **Grounded generation:** the AI may only use your material. Every FAQ carries a quote copied from it, and code checks that the quote exists.
- **Fact checks:** numbers, prices, times, emails and web addresses in an answer must appear in your material.
- **Needs review:** FAQs that fail a check start unticked. Nothing unverified is exported until you approve it.
- **Urdu output:** a faithful translation of the English text, in Urdu script, with numbers kept the same. Code checks that numbers, emails and links match and that the text really is Urdu.
- **Editing:** edit any cell in a table, edit one FAQ in full text boxes with a right-to-left Urdu preview, add rows, delete rows, re-translate after changing the English.
- **Add from a question:** type a customer question; it is answered only if your material covers it, otherwise the app says so.
- **Missing information:** suggests questions customers may ask that your material does not answer.
- **Export:** a mobile-friendly web page (English, Urdu or both, Urdu right-to-left, with FAQ markup for search engines), CSV (opens in Excel with Urdu), JSON and plain text.

## Important limits
- The checks catch made-up quotes and numbers that are not in your material. They cannot tell if a real number is used in the wrong place. **Read each answer against your material.**
- AI Urdu can contain mistakes. **Have a native Urdu speaker read it before you publish.**
- Numbers written as words ("thirty days") are not checked.
- Your material is sent to Groq to write the FAQs. Do not upload secrets or customers' personal data.
- Scanned PDFs (images of pages) have no text and will not work.

## Files
| File | Purpose |
|---|---|
| `app.py` | The whole Streamlit app |
| `requirements.txt` | Python packages (streamlit, pandas, pypdf, python-docx, groq) |
| `.gitignore` | Keeps secrets and private documents out of GitHub |

## Groq API key (required)
1. Create a free key at https://console.groq.com (API Keys → Create API Key).
2. On Streamlit Community Cloud: app **Settings → Secrets**, paste one line with straight quotes:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
3. Save and **Reboot app**.

Locally, put the same line in `.streamlit/secrets.toml` (already in `.gitignore`). Never put the key in `app.py`.

## Try it
Click **Use sample business (Sunrise Bakery)**, then **Generate FAQs**. You should get about 10 FAQs with source quotes.
Then try **Add an FAQ from your own question** with "Do you make eggless cakes?". The sample material does not mention it, so the app should
refuse instead of inventing an answer.

## Run on your computer (optional)
```
pip install -r requirements.txt
streamlit run app.py
```

## Storage note
No database is needed. To keep FAQs between visits, use a free hosted database such as Neon PostgreSQL;
Streamlit Community Cloud erases local files (including SQLite) when the app restarts. Download your FAQs before you leave.
