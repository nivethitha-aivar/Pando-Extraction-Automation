"""
DSPy-based Invoice Extraction System
Automatically learns to extract structured invoice data from Textract JSON
without manually writing prompts for each carrier.

THE GOAL:
---------
1. Input: Textract JSON (from AWS Textract) - e.g., from textract_input.py or actual Textract output
2. Output: Structured payload matching output_paylod.json schema
3. DSPy automatically creates the prompt by learning from examples

HOW IT WORKS:
-------------
- You provide: Textract JSON + Target Output (from textract_input.py MODEL_OUTPUT)
- DSPy learns: How to map Textract fields to your schema
- DSPy generates: Optimal prompts automatically
- Result: Works with new carriers without manual prompt writing

Configured with AWS Bedrock:
- Model: bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0
- Region: us-east-1 (configurable via REGION env var)

Usage:
    # Simple: Just run with your Textract JSON file
    export TEXTRACT_JSON_FILE="path/to/textract_output.json"
    python python.py
    
    # Or place textract_output.json in current directory
    python python.py
    
    # Programmatic:
    from python import configure_dspy, compile_invoice_extractor, create_examples_from_data
    from textract_input import MODEL_OUTPUT
    
    configure_dspy()
    textract_data = load_textract_json("textract_output.json")
    examples = create_examples_from_data(textract_data, MODEL_OUTPUT)
    extractor = compile_invoice_extractor(examples)
"""

import dspy
import json
from typing import Dict, Any, List
from dspy.teleprompt import MIPROv2
from dspy import Example
import os
import boto3
from botocore.config import Config
from datetime import datetime


# ============================================================================
# CONFIGURATION: Bedrock Model Setup
# ============================================================================

# Bedrock model configuration
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0')
REGION = os.environ.get('REGION', 'us-east-1')

# Custom Bedrock LM class for DSPy
class BedrockLM(dspy.LM):
    """Custom DSPy LM class for AWS Bedrock."""
    
    def __init__(self, model_id=BEDROCK_MODEL_ID, region=REGION):
        self.model_id = model_id
        self.region = region
        self._bedrock = None  # Lazy initialization for pickling
        super().__init__(model_id)
    
    @property
    def bedrock(self):
        """Lazy initialization of Bedrock client to avoid pickling issues."""
        if self._bedrock is None:
            self._bedrock = boto3.client(
                'bedrock-runtime', 
                region_name=self.region,
                config=Config(connect_timeout=30, read_timeout=400)
            )
        return self._bedrock
    
    def __getstate__(self):
        """Make the class pickleable by excluding the boto3 client."""
        state = self.__dict__.copy()
        # Remove the boto3 client - it will be recreated on first access
        state['_bedrock'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # Client will be recreated lazily on first access
        self._bedrock = None
    
    def basic_request(self, prompt, **kwargs):
        """Make a request to Bedrock."""
        try:
            response = self.bedrock.converse(
                modelId=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt}]
                }],
                additionalModelRequestFields={
                    "max_tokens": kwargs.get("max_tokens", 4000)
                }
            )
            
            # Extract text from response
            content = response.get("output", {}).get("message", {}).get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "")
            return ""
        except Exception as e:
            print(f"❌ BedrockLM request failed: {e}")
            raise


# ============================================================================
# STEP 1: Define the Signature (The Contract - Never Changes)
# ============================================================================

class InvoiceExtraction(dspy.Signature):
    """
    Extract structured invoice data from raw AWS Textract output.
    
    Map varied field names and layouts (like 'Inv #', 'Invoice No', 'Ref', 
    different positions) to the standard schema defined in output_paylod.json.
    
    Handle different carrier formats (Robinson, Meta, etc.) by learning from examples.
    Dates must be formatted as YYYY-MM-DD or ISO 8601 format.
    Empty fields should be returned as empty strings "".
    """
    
    # The Input: Raw messy dump from AWS Textract
    textract_json = dspy.InputField(
        desc="Raw JSON output from AWS Textract containing Blocks array with Text, Geometry, Relationships, etc."
    )
    
    # The Output: Perfect, clean JSON format matching output_paylod.json schema
    structured_payload = dspy.OutputField(
        desc="Normalized JSON matching the output_paylod.json schema with keys: invoice_number, invoice_date, payment_due_date, vendor_reference_id, currency, total_invoice_value, total_tax_amount, bill_of_lading_number, bill_to_name, bill_to_gst, bill_to_address, bill_to_phone_number, bill_to_email, cost_center, project_code, billing_entity_name, shipments (array with shipment_number, mode, pro_number, source_name, destination_name, source_city, source_state, source_country, source_zip_code, destination_city, destination_state, destination_country, destination_zip_code, shipment_weight, shipment_weight_uom, shipment_total_value, charges, etc.)"
    )


# ============================================================================
# STEP 2: Helper Functions for Data Conversion
# ============================================================================

def convert_model_output_to_target_schema(model_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MODEL_OUTPUT format (with value/explanation/confidence) 
    to the target schema format (just values, matching output_paylod.json).
    """
    def extract_value(obj):
        """Recursively extract 'value' from nested structures."""
        if isinstance(obj, dict):
            if 'value' in obj:
                val = obj['value']
                # Recursively process nested structures
                if isinstance(val, dict):
                    return {k: extract_value(v) for k, v in val.items()}
                elif isinstance(val, list):
                    return [extract_value(item) for item in val]
                else:
                    # Convert None to empty string for string fields
                    if val is None:
                        return ""
                    return val
            else:
                # No 'value' key, process all keys
                return {k: extract_value(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [extract_value(item) for item in obj]
        else:
            # Convert None to empty string
            if obj is None:
                return ""
            return obj
    
    result = extract_value(model_output)
    
    # Ensure shipments array structure matches output_paylod.json
    if 'shipments' in result and isinstance(result['shipments'], list):
        for shipment in result['shipments']:
            # Add missing fields from output_paylod.json schema
            shipment.setdefault('source_code', '')
            shipment.setdefault('source_zone', '')
            shipment.setdefault('destination_code', '')
            shipment.setdefault('destination_zone', '')
            shipment.setdefault('taxes', [])
            shipment.setdefault('custom_charges', [])
            
            # Handle custom field - map number_of_pallets if it exists at shipment level
            if 'number_of_pallets' in shipment:
                no_of_pallets = shipment.pop('number_of_pallets')
            else:
                no_of_pallets = ''
            
            shipment.setdefault('custom', {
                'reference_number': '',
                'special_instructions': '',
                'priority': '',
                'pay_as_present': '',
                'gl_code': '',
                'no_of_pallets': no_of_pallets if no_of_pallets != '' else ''
            })
            
            shipment.setdefault('shipment_identifiers', {
                'booking_number': '',
                'container_numbers': ''
            })
            
            # Handle container_numbers from container array
            if 'container' in shipment and isinstance(shipment['container'], list):
                container_ids = []
                for container in shipment['container']:
                    if isinstance(container, dict) and 'container_id' in container:
                        container_ids.append(str(container['container_id']))
                if container_ids:
                    shipment['shipment_identifiers']['container_numbers'] = ', '.join(container_ids)
                # Remove container field as it's not in the target schema
                shipment.pop('container', None)
            
            # Ensure charges array has all required fields
            if 'charges' in shipment and isinstance(shipment['charges'], list):
                for charge in shipment['charges']:
                    # Remove fields not in target schema
                    charge.pop('unit', None)
                    charge.pop('pieces', None)
                    # Ensure required fields exist
                    charge.setdefault('charge_code', '')
                    charge.setdefault('charge_name', '')
                    charge.setdefault('charge_gross_amount', '')
                    charge.setdefault('charge_tax_amount', '')
                    charge.setdefault('currency', '')
            
            # Remove fields not in target schema
            shipment.pop('port_of_loading', None)
            shipment.pop('port_of_discharge', None)
            shipment.pop('dangerous_goods_indicator', None)
            shipment.pop('source_state_code', None)
            shipment.pop('destination_state_code', None)
    
    # Handle delivery_type - move from root to shipments if needed
    delivery_type = result.pop('delivery_type', None)
    
    # Remove additional_info if not in target schema
    result.pop('additional_info', None)
    result.pop('payment_terms', None)
    
    # Ensure delivery_type is in shipments (move from root if it was there)
    if 'shipments' in result and isinstance(result['shipments'], list):
        for shipment in result['shipments']:
            if delivery_type and 'delivery_type' not in shipment:
                shipment['delivery_type'] = delivery_type
            shipment.setdefault('delivery_type', '')
            shipment.setdefault('service_level', '')
    
    return result


def load_textract_json(file_path: str) -> Dict[str, Any]:
    """Load Textract JSON from file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_textract_from_file_or_env(file_path: str = None) -> Dict[str, Any]:
    """
    Load actual Textract JSON from file.
    
    In production, this should be the actual Textract output from AWS.
    You can either:
    1. Pass a file path directly
    2. Set TEXTRACT_JSON_FILE environment variable
    3. The function will look for common file names
    
    Args:
        file_path: Path to Textract JSON file (optional)
    
    Returns:
        Textract JSON dict with Blocks array
    """
    # Try multiple sources
    possible_paths = []
    
    if file_path:
        possible_paths.append(file_path)
    
    # Check environment variable
    env_path = os.environ.get('TEXTRACT_JSON_FILE')
    if env_path:
        possible_paths.append(env_path)
    
    # Common file names
    possible_paths.extend([
        'textract_output.json',
        'textract.json',
        'textract_dump.json',
        'dump.json'
    ])
    
    # Try to load from any available path
    for path in possible_paths:
        if os.path.isfile(path):
            print(f"📄 Loading Textract JSON from: {path}")
            return load_textract_json(path)
    
    # If no file found, raise error
    raise FileNotFoundError(
        f"No Textract JSON file found. Please provide one of:\n"
        f"  - Pass file_path parameter\n"
        f"  - Set TEXTRACT_JSON_FILE environment variable\n"
        f"  - Place file in current directory with name: textract_output.json\n"
        f"Tried paths: {possible_paths}"
    )


# ============================================================================
# STEP 3: Create Training Examples
# ============================================================================

def parse_example_outputs_from_log(log_file: str) -> List[Dict[str, Any]]:
    """
    Parse MODEL_OUTPUT dictionaries from example.py log file.
    
    Args:
        log_file: Path to example.py log file
        
    Returns:
        List of MODEL_OUTPUT dictionaries
    """
    outputs = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            file_content = f.read()
        
        # Find all occurrences of "Extracted info from meta.py: "
        import re
        pattern = r"Extracted info from meta\.py:\s*"
        match_positions = [m.end() for m in re.finditer(pattern, file_content)]
        
        for start_pos in match_positions:
            # Find the matching closing brace by counting braces
            brace_count = 0
            i = start_pos
            dict_start = start_pos
            
            while i < len(file_content):
                char = file_content[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # Found the end of the dictionary
                        dict_str = file_content[dict_start:i+1]
                        try:
                            output_dict = eval(dict_str)
                            outputs.append(output_dict)
                        except Exception as e:
                            pass
                        break
                i += 1
        
        return outputs
    except Exception as e:
        print(f"   ⚠️  Could not parse {log_file}: {e}")
        return []


def create_examples_from_data(
    textract_data: str,  # Changed: Now accepts raw text string
    target_output: Dict[str, Any]
) -> List[Example]:
    """
    Create DSPy Examples from raw text input and target output.
    
    This is the KEY function: It creates training examples that teach DSPy:
    "When you see raw text like THIS, extract structured payload like THAT"
    
    Args:
        textract_data: Raw text string (from FREIGHT_INVOICE_TEXT + BILL_OF_LADING_TEXT) - THE INPUT
        target_output: Target structured payload (MODEL_OUTPUT dict with value/explanation/confidence) - THE DESIRED OUTPUT
    
    Returns:
        List of DSPy Examples (input-output pairs for learning)
    """
    # textract_data is now a raw text string (not JSON)
    textract_str = textract_data if isinstance(textract_data, str) else str(textract_data)
    
    # target_output should be MODEL_OUTPUT format (dict with value/explanation/confidence)
    # Keep it as-is - don't convert, DSPy should learn to produce this format
    target = target_output if isinstance(target_output, dict) else target_output
    
    # Convert to JSON strings for DSPy
    target_str = json.dumps(target, indent=2)
    
    # Create DSPy Example
    example = Example(
        textract_json=textract_str,  # Raw text input
        structured_payload=target_str  # MODEL_OUTPUT format output
    ).with_inputs("textract_json")
    
    return [example]


# ============================================================================
# STEP 4: Validation Metric
# ============================================================================

def validate_json_match(example, pred, trace=None):
    """
    Validate that the predicted JSON matches the expected output.
    This is a simple exact match - you can make it more sophisticated.
    """
    try:
        expected = json.loads(example.structured_payload)
        predicted = json.loads(pred.structured_payload)
        
        # Simple comparison - in production, you might want field-level comparison
        return expected == predicted
    except:
        return False


# ============================================================================
# STEP 5: Main Training/Compilation Function
# ============================================================================

def compile_invoice_extractor(
    examples: List[Example],
    output_path: str = "optimized_invoice_extractor.json",
    max_bootstrapped_demos: int = 4
) -> dspy.Module:
    """
    Compile the invoice extractor using MIPROv2 optimizer.
    
    Args:
        examples: List of DSPy Examples (input-output pairs)
        output_path: Path to save the compiled program
        max_bootstrapped_demos: Maximum number of examples to include in prompt
    
    Returns:
        Compiled DSPy Module
    """
    print(f"🚀 Starting DSPy compilation with {len(examples)} examples...")
    
    # MIPROv2 requires at least 2 examples - duplicate if needed
    if len(examples) < 2:
        print(f"   ⚠️  MIPROv2 requires at least 2 examples. Creating a duplicate...")
        # Create a copy of the example
        original = examples[0]
        duplicate = Example(
            textract_json=original.textract_json,
            structured_payload=original.structured_payload
        ).with_inputs("textract_json")
        examples = examples + [duplicate]
        print(f"   ✅ Now using {len(examples)} examples (1 original + 1 duplicate)")
    
    # Split into train and validation sets (80/20 split, minimum 1 in each)
    split_idx = max(1, int(len(examples) * 0.8))
    trainset = examples[:split_idx]
    valset = examples[split_idx:] if split_idx < len(examples) else examples[:1]
    
    print(f"   Training examples: {len(trainset)}, Validation examples: {len(valset)}")
    
    # Initialize the optimizer
    teleprompter = MIPROv2(
        metric=validate_json_match,
        max_bootstrapped_demos=max_bootstrapped_demos
    )
    
    # Create the base module
    base_module = dspy.ChainOfThought(InvoiceExtraction)
    
    # Compile (this is where DSPy "writes" the prompt)
    print("📝 DSPy is analyzing examples and generating optimal prompts...")
    print("   DSPy will:")
    print("   1. Look at your Textract JSON examples")
    print("   2. Look at your target output payload examples")
    print("   3. Learn the mapping between them")
    print("   4. Generate prompts that extract data correctly")
    print("   5. Test different prompt variations")
    print("   6. Select the best one")
    print()
    
    # Capture logs to extract generated prompts
    import logging
    import io
    import sys
    
    # Set up logging capture
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    dspy_logger = logging.getLogger('dspy.teleprompt.mipro_optimizer_v2')
    dspy_logger.addHandler(handler)
    dspy_logger.setLevel(logging.INFO)
    
    optimized_extractor = teleprompter.compile(
        student=base_module,
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=max_bootstrapped_demos
    )
    
    # Extract and display generated prompts from logs
    log_content = log_capture.getvalue()
    dspy_logger.removeHandler(handler)
    
    # Extract proposed instructions from logs
    print("\n" + "="*80)
    print("📋 DSPy GENERATED PROMPTS (Automatically Created)")
    print("="*80)
    
    import re
    # Find all proposed instructions
    instruction_pattern = r'INFO dspy\.teleprompt\.mipro_optimizer_v2: (\d+): (.*?)(?=INFO dspy\.teleprompt\.mipro_optimizer_v2: \d+:|==> STEP|$)'
    instructions = re.findall(instruction_pattern, log_content, re.DOTALL)
    
    all_prompts = {}
    
    if instructions:
        print(f"\n✅ DSPy Generated {len(instructions)} Instruction Candidates:\n")
        for num, instruction in instructions:
            print("─" * 80)
            print(f"INSTRUCTION CANDIDATE #{num}")
            print("─" * 80)
            # Clean up the instruction text
            clean_instruction = instruction.strip()
            # Remove extra whitespace
            clean_instruction = re.sub(r'\n{3,}', '\n\n', clean_instruction)
            print(clean_instruction)
            print()
            all_prompts[f"candidate_{num}"] = clean_instruction
    else:
        # Try to find in the log content directly
        if "Proposed Instructions" in log_content:
            start_idx = log_content.find("Proposed Instructions")
            section = log_content[start_idx:start_idx+10000]
            for i in range(3):
                pattern = f'{i}: (.*?)(?={i+1}: |==> STEP|$)'
                match = re.search(pattern, section, re.DOTALL)
                if match:
                    print("─" * 80)
                    print(f"INSTRUCTION CANDIDATE #{i}")
                    print("─" * 80)
                    prompt_text = match.group(1).strip()
                    print(prompt_text)
                    print()
                    all_prompts[f"candidate_{i}"] = prompt_text
    
    # Save all prompts to a file
    prompts_file = "dspy_generated_prompts.json"
    try:
        final_instruction = ""
        if hasattr(optimized_extractor, 'predictor'):
            predictor = optimized_extractor.predictor
            if hasattr(predictor, 'signature'):
                sig = predictor.signature
                if hasattr(sig, 'instructions'):
                    final_instruction = sig.instructions
        elif hasattr(optimized_extractor, 'signature'):
            sig = optimized_extractor.signature
            if hasattr(sig, 'instructions'):
                final_instruction = sig.instructions
        
        prompts_data = {
            "all_candidates": all_prompts,
            "final_selected": final_instruction,
            "compilation_date": str(datetime.now())
        }
        
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(prompts_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 All prompts saved to: {prompts_file}")
        print()
    except Exception as e:
        print(f"   ⚠️  Could not save prompts to file: {e}")
    
    # Show the final selected instruction from the compiled extractor
    print("="*80)
    print("✅ FINAL SELECTED PROMPT (Saved in compiled extractor)")
    print("="*80)
    try:
        # Access the compiled module's signature
        if hasattr(optimized_extractor, 'predictor'):
            predictor = optimized_extractor.predictor
            if hasattr(predictor, 'signature'):
                sig = predictor.signature
                if hasattr(sig, 'instructions'):
                    print("\n" + sig.instructions)
                    print()
        elif hasattr(optimized_extractor, 'signature'):
            sig = optimized_extractor.signature
            if hasattr(sig, 'instructions'):
                print("\n" + sig.instructions)
                print()
        else:
            # Load from saved file
            with open(output_path, 'r') as f:
                saved = json.load(f)
                if 'predict' in saved and 'signature' in saved['predict']:
                    instructions = saved['predict']['signature'].get('instructions', '')
                    if instructions:
                        print("\n" + instructions)
                        print()
    except Exception as e:
        print(f"   ⚠️  Could not extract final prompt: {e}")
    
    print("="*80)
    print()
    
    # Save the compiled program
    optimized_extractor.save(output_path)
    print(f"✅ Compiled program saved to {output_path}")
    
    return optimized_extractor


# ============================================================================
# STEP 6: Usage Function
# ============================================================================

def extract_invoice(extractor: dspy.Module, textract_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use the compiled extractor to extract invoice data from new Textract JSON.
    
    Args:
        extractor: Compiled DSPy Module
        textract_json: Raw Textract JSON (dict or file path)
    
    Returns:
        Structured invoice payload
    """
    # Handle file path
    if isinstance(textract_json, str) and os.path.isfile(textract_json):
        textract_data = load_textract_json(textract_json)
    else:
        textract_data = textract_json
    
    # Convert to JSON string
    textract_str = json.dumps(textract_data, indent=2)
    
    # Run extraction
    result = extractor(textract_json=textract_str)
    
    # Parse result - handle markdown code blocks
    output = result.structured_payload
    
    # Remove markdown code blocks if present
    import re
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', output, re.DOTALL)
    if json_match:
        output = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'(\{.*\})', output, re.DOTALL)
        if json_match:
            output = json_match.group(1)
    
    # Parse JSON
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        # Try to fix common issues
        try:
            # Remove trailing commas
            output = re.sub(r',\s*}', '}', output)
            output = re.sub(r',\s*]', ']', output)
            return json.loads(output)
        except:
            return {"error": "Failed to parse output", "raw_output": result.structured_payload, "parse_error": str(e)}


# ============================================================================
# STEP 7: Main Execution
# ============================================================================

def configure_dspy(model_id=None, region=None):
    """
    Configure DSPy with Bedrock model.
    
    Args:
        model_id: Bedrock model ID (default: uses BEDROCK_MODEL_ID env var or default)
                 Example: 'bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0'
        region: AWS region (default: uses REGION env var or 'us-east-1')
    
    Returns:
        True if configuration successful
    
    Example:
        configure_dspy(model_id='bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0')
    """
    global BEDROCK_MODEL_ID, REGION
    
    if model_id:
        BEDROCK_MODEL_ID = model_id
    if region:
        REGION = region
    
    print(f"🔧 Configuring DSPy with Bedrock model: {BEDROCK_MODEL_ID}")
    print(f"   Region: {REGION}")
    
    try:
        lm = BedrockLM(model_id=BEDROCK_MODEL_ID, region=REGION)
        dspy.configure(lm=lm)
        print("✅ DSPy configured successfully!")
        print()
        return True
    except Exception as e:
        print(f"❌ Failed to configure DSPy: {e}")
        print()
        print("💡 Make sure:")
        print("   1. AWS credentials are configured (aws configure)")
        print("   2. You have permissions to use Bedrock")
        print("   3. The model ID is correct")
        raise


def main():
    """
    Main function demonstrating the workflow:
    1. Configure DSPy with Bedrock
    2. Load example data (Textract input + Target output)
    3. Create DSPy Examples
    4. Compile the extractor
    5. Test on new data
    """
    
    # Configure DSPy with Bedrock
    configure_dspy()
    
    # Load your data
    print("📂 Loading example data...")
    print()
    print("   DSPy Learning Process:")
    print("   ┌─────────────────────────────────────────┐")
    print("   │  Input:  textract_input.py (Raw text)  │")
    print("   │     ↓                                    │")
    print("   │  DSPy learns the extraction             │")
    print("   │     ↓                                    │")
    print("   │  Output: example.py (MODEL_OUTPUT)       │")
    print("   └─────────────────────────────────────────┘")
    print()
    
    # Load RAW TEXT from textract_input.py (FREIGHT_INVOICE_TEXT + BILL_OF_LADING_TEXT)
    from textract_input import FREIGHT_INVOICE_TEXT, BILL_OF_LADING_TEXT
    print("   ✅ Loaded RAW TEXT from textract_input.py")
    print("      - FREIGHT_INVOICE_TEXT")
    print("      - BILL_OF_LADING_TEXT")
    
    # Combine both texts as the input (this is what Textract extracted)
    combined_text = f"FREIGHT INVOICE:\n{FREIGHT_INVOICE_TEXT}\n\nBILL OF LADING:\n{BILL_OF_LADING_TEXT}"
    textract_input = combined_text  # Raw text input
    print("   ✅ Combined raw text (this is the INPUT for DSPy)")
    print("      (DSPy will learn to extract structured data from this raw text)")
    
    # Parse target outputs from example.py log file
    print()
    print("   📂 Loading target outputs from example.py...")
    target_outputs = parse_example_outputs_from_log("example.py")
    
    if not target_outputs:
        print("   ❌ ERROR: Could not find target outputs in example.py!")
        print("      Please ensure example.py contains log entries with 'Extracted info from meta.py:'")
        print("      Example format:")
        print("      2025-11-26 05:31:35,616 - INFO - Extracted info from meta.py: {'invoice_number': ...}")
        raise ValueError("No target outputs found for training")
    else:
        print(f"   ✅ Found {len(target_outputs)} target output(s) in example.py")
        print("      (These show DSPy what the correct extracted output should look like)")
        print("      (Matches output_paylod.json schema with value/explanation/confidence)")
    
    # Create examples using first target output with textract_input.py input
    print()
    print("📝 Creating training examples...")
    print("   Each example teaches DSPy:")
    print("   'When I give you raw text like textract_input.py → extract to MODEL_OUTPUT format like example.py'")
    
    examples = []
    # Use textract_input.py input with first target output
    examples.extend(create_examples_from_data(textract_input, target_outputs[0]))
    
    # If there are more target outputs but we only have one input, we can still use them
    # (DSPy will learn the output format even if inputs are similar)
    if len(target_outputs) > 1:
        print(f"   ℹ️  Found {len(target_outputs)} different carrier outputs")
        print("      Using first output with textract_input.py input")
        print("      (To use all outputs, provide corresponding input texts for each)")
    
    print(f"✅ Created {len(examples)} example(s)")
    print()
    
    # Add more examples as you get them (for different carriers)
    # meta_textract = load_textract_json("meta_textract.json")
    # meta_target = load_textract_json("meta_output.json")
    # meta_examples = create_examples_from_data(meta_textract, meta_target)
    # examples.extend(meta_examples)
    
    # Compile the extractor
    print("🔧 Compiling invoice extractor...")
    print("   (This may take a few minutes as DSPy optimizes the prompts)")
    print()
    
    try:
        extractor = compile_invoice_extractor(
            examples=examples,
            output_path="optimized_invoice_extractor.json",
            max_bootstrapped_demos=4
        )
        
        print()
        print("🎉 Compilation complete!")
        print()
        print("📋 Usage:")
        print("   from python import extract_invoice, load_textract_json")
        print("   extractor = dspy.load('optimized_invoice_extractor.json')")
        print("   result = extract_invoice(extractor, 'new_textract_output.json')")
        print()
        
    except Exception as e:
        print(f"❌ Error during compilation: {e}")
        print()
        print("💡 Make sure:")
        print("   1. DSPy is configured (should be done automatically)")
        print("   2. AWS credentials are set up correctly")
        print("   3. You have permissions to use Bedrock")
        print("   4. The model ID is correct:", BEDROCK_MODEL_ID)
        raise


if __name__ == "__main__":
    main()

