## Troubleshooting

### Error: Can't Find spaCy Model

If you encounter an error like `OSError: [E050] Can't find model 'en_core_web_md'`, follow these steps to resolve it:

1. Make sure you have installed spaCy and the required model by running:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
