"""
Test extraction using textract_input.py raw text data
This tests the compiled DSPy extractor on new data.
"""

import dspy
import json
from python import configure_dspy, extract_invoice
from textract_input import FREIGHT_INVOICE_TEXT, BILL_OF_LADING_TEXT

# Configure DSPy
print("🔧 Configuring DSPy...")
configure_dspy()

# Load compiled extractor
print("📂 Loading compiled extractor...")
try:
    extractor = dspy.load("optimized_invoice_extractor.json")
    print("✅ Extractor loaded!")
except Exception as e:
    print(f"⚠️  Could not load compiled extractor: {e}")
    print("   Using base extractor instead...")
    from python import InvoiceExtraction
    extractor = dspy.ChainOfThought(InvoiceExtraction)
    print("⚠️  Using base extractor (compiled one not found)")

# Prepare input: Combine raw text from textract_input.py
print("\n📝 Preparing input data...")
combined_text = f"FREIGHT INVOICE:\n{FREIGHT_INVOICE_TEXT}\n\nBILL OF LADING:\n{BILL_OF_LADING_TEXT}"
print("   ✅ Loaded raw text from textract_input.py")
print("      - FREIGHT_INVOICE_TEXT")
print("      - BILL_OF_LADING_TEXT")

# Test extraction
print("\n🚀 Testing extraction...")
print("   Input: textract_input.py (Raw text from FREIGHT_INVOICE_TEXT + BILL_OF_LADING_TEXT)")
print("   Expected Output: MODEL_OUTPUT format (with value/explanation/confidence)")
print()

result = extract_invoice(extractor, combined_text)

print("\n✅ Extraction complete!")
print("\n" + "="*60)
print("EXTRACTED DATA (MODEL_OUTPUT format):")
print("="*60)
print(json.dumps(result, indent=2))

# Save result
with open("dump_extraction_result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n💾 Result saved to: dump_extraction_result.json")

print("\n" + "="*60)
print("📋 Next Steps:")
print("="*60)
print("1. Review the extracted data in dump_extraction_result.json")
print("2. Compare it with the target output from example.py")
print("3. If results are good, you can use this extractor on new invoices!")
print("4. To extract from a new invoice:")
print("   - Prepare raw text (FREIGHT_INVOICE_TEXT + BILL_OF_LADING_TEXT)")
print("   - Call: extract_invoice(extractor, raw_text_string)")
print("="*60)
