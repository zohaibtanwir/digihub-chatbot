# Service Line Headers

Place JSON files with headers for each service line here.

## File Naming Convention
- Use lowercase with underscores matching service line enum names
- Examples: `airport_solutions.json`, `world_tracer.json`, `bag_manager.json`

## JSON Structure
Each file should contain an array of heading objects:
```json
[
  {"heading": "Header Text 1"},
  {"heading": "Header Text 2"}
]
```

## Supported Service Lines
See `/src/enums/subscriptions.py` for the complete list of service lines:
- airport_solutions → Airport Solutions
- world_tracer → World Tracer
- bag_manager → Bag Manager
- billing → Billing
- operational_support → Operational Support
- community_messaging_kb → Community Messaging KB
- airport_committee → Airport Committee
- euro_customer_advisory_board → Euro Customer Advisory Board
- aero_performance → Aero Performance
- apac_customer_advisory_board → APAC Customer Advisory Board
- general_info → General Info

## Usage
1. Add your JSON files to this directory
2. Run the keyword extraction script: `python src/scripts/extract_service_line_keywords.py`
3. The script will generate `/src/data/service_line_keywords.json` with extracted keywords
