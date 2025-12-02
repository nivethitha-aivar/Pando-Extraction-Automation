# Pando Extraction Automation

An intelligent invoice extraction system powered by **DSPy** and **AWS Bedrock** that automatically learns to extract structured invoice data from AWS Textract output without manually writing prompts for each carrier.

## 🎯 Overview

This project uses **DSPy** (Declarative Self-improving Python) to automatically generate optimal prompts for extracting structured invoice data from raw Textract output. Instead of manually crafting prompts for each carrier format, the system learns from examples and generates the best extraction prompts automatically.

### Key Features

- 🤖 **Automatic Prompt Generation**: DSPy learns from examples and creates optimal extraction prompts
- 🔄 **Carrier-Agnostic**: Works with different carrier formats without manual prompt writing
- 📊 **Structured Output**: Extracts data matching the `output_paylod.json` schema
- ☁️ **AWS Bedrock Integration**: Uses Claude Haiku model via AWS Bedrock
- 🎓 **Learning-Based**: Improves extraction accuracy by learning from training examples

## 🏗️ How It Works

```
┌─────────────────────────────────────────┐
│  Input: textract_input.py (Raw text)   │
│     ↓                                    │
│  DSPy learns the extraction             │
│     ↓                                    │
│  Output: Structured JSON (MODEL_OUTPUT) │
└─────────────────────────────────────────┘
```

1. **Input**: Raw text from AWS Textract (freight invoices, bills of lading)
2. **Learning**: DSPy analyzes input-output pairs to understand extraction patterns
3. **Optimization**: MIPROv2 optimizer generates and tests multiple prompt variations
4. **Output**: Structured JSON matching your target schema

## 📋 Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- AWS CLI configured with appropriate credentials
- Access to the following AWS Bedrock model:
  - `bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0`

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nivethitha-aivar/Pando-Extraction-Automation.git
   cd Pando-Extraction-Automation
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS credentials**
   ```bash
   aws configure
   ```
   
   Or set environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   ```

4. **Configure Bedrock model (optional)**
   ```bash
   export BEDROCK_MODEL_ID="bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0"
   export REGION="us-east-1"
   ```

## 📁 Project Structure

```
Pando-Extraction-Automation/
├── python.py              # Main DSPy extraction system
├── extraction.py          # Test extraction script
├── textract_input.py      # Input data (FREIGHT_INVOICE_TEXT, BILL_OF_LADING_TEXT)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── optimized_invoice_extractor.json  # Compiled extractor (generated)
```

## 🔧 Configuration

### Environment Variables

- `BEDROCK_MODEL_ID`: Bedrock model identifier (default: `bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0`)
- `REGION`: AWS region (default: `us-east-1`)
- `TEXTRACT_JSON_FILE`: Path to Textract JSON file (optional)

## 💻 Usage

### 1. Training/Compiling the Extractor

Train the extractor using example data:

```bash
python python.py
```

This will:
- Load raw text from `textract_input.py`
- Load target outputs from `example.py` (if available)
- Create training examples
- Compile the extractor using MIPROv2 optimizer
- Save the compiled extractor to `optimized_invoice_extractor.json`

### 2. Testing Extraction

Test the compiled extractor on new data:

```bash
python extraction.py
```

This will:
- Load the compiled extractor
- Extract data from `textract_input.py`
- Save results to `dump_extraction_result.json`

### 3. Programmatic Usage

```python
from python import configure_dspy, extract_invoice, compile_invoice_extractor, create_examples_from_data
from textract_input import FREIGHT_INVOICE_TEXT, BILL_OF_LADING_TEXT
import dspy

# Configure DSPy
configure_dspy()

# Load compiled extractor
extractor = dspy.load("optimized_invoice_extractor.json")

# Prepare input text
combined_text = f"FREIGHT INVOICE:\n{FREIGHT_INVOICE_TEXT}\n\nBILL OF LADING:\n{BILL_OF_LADING_TEXT}"

# Extract invoice data
result = extract_invoice(extractor, combined_text)

# Result is a structured dictionary matching output_paylod.json schema
print(result)
```

### 4. Creating Training Examples

```python
from python import create_examples_from_data
from textract_input import FREIGHT_INVOICE_TEXT, BILL_OF_LADING_TEXT

# Your target output (MODEL_OUTPUT format)
target_output = {
    "invoice_number": {"value": "04225060041", "explanation": "...", "confidence": 0.95},
    # ... rest of the structure
}

# Create training example
combined_text = f"FREIGHT INVOICE:\n{FREIGHT_INVOICE_TEXT}\n\nBILL OF LADING:\n{BILL_OF_LADING_TEXT}"
examples = create_examples_from_data(combined_text, target_output)

# Compile extractor
extractor = compile_invoice_extractor(examples)
```

## 📊 Output Format

The extractor produces structured JSON matching the `output_paylod.json` schema:

```json
{
  "invoice_number": "04225060041",
  "invoice_date": "2025-10-31",
  "currency": "USD",
  "total_invoice_value": "21651.14",
  "shipments": [
    {
      "shipment_number": "...",
      "mode": "...",
      "source_name": "...",
      "destination_name": "...",
      "charges": [...]
    }
  ]
}
```

## 🔍 Key Components

### `python.py`
Main DSPy extraction system containing:
- `BedrockLM`: Custom DSPy language model for AWS Bedrock
- `InvoiceExtraction`: DSPy signature defining input/output contract
- `compile_invoice_extractor()`: Compiles extractor using MIPROv2 optimizer
- `extract_invoice()`: Extracts invoice data from raw text
- `configure_dspy()`: Configures DSPy with Bedrock model

### `extraction.py`
Test script that:
- Loads compiled extractor
- Tests extraction on sample data
- Saves results to JSON file

### `textract_input.py`
Contains sample input data:
- `FREIGHT_INVOICE_TEXT`: Raw freight invoice text
- `BILL_OF_LADING_TEXT`: Raw bill of lading text

## 🎓 How DSPy Works

1. **Signature Definition**: Defines what goes in and what comes out
2. **Example Creation**: Creates input-output pairs for training
3. **Optimization**: MIPROv2 tests different prompt variations
4. **Selection**: Chooses the best performing prompt
5. **Compilation**: Saves the optimized extractor for reuse

## 📝 Notes

- The compiled extractor (`optimized_invoice_extractor.json`) contains the learned prompts
- Generated prompts are saved to `dspy_generated_prompts.json`
- The system automatically handles different carrier formats by learning from examples
- Empty fields are returned as empty strings `""`
- Dates are formatted as YYYY-MM-DD or ISO 8601

## 🐛 Troubleshooting

### "Could not load compiled extractor"
- Run `python.py` first to compile the extractor
- Ensure `optimized_invoice_extractor.json` exists

### "Failed to configure DSPy"
- Check AWS credentials: `aws configure`
- Verify Bedrock model access
- Check region configuration

### "No target outputs found"
- Ensure `example.py` contains log entries with `Extracted info from meta.py:`
- Or provide target outputs programmatically

## 📄 License

[Add your license here]

## 👥 Contributors

- [Your Name/Team]

## 🔗 Related Resources

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Textract Documentation](https://docs.aws.amazon.com/textract/)

---

**Note**: This project is designed for invoice extraction automation. Ensure you have proper permissions and comply with data privacy regulations when processing invoices.

