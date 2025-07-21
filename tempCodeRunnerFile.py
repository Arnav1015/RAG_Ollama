When the query asks for creating excel sheet, ALWAYS format your response as a list of dictionaries that can be exported to Excel.

EXCEL FORMATTING RULES:
1. ALWAYS wrap structured data in triple backticks with python syntax highlighting:
```python
[
    {{"column1": "value1", "column2": "value2", ...}},
    {{"column1": "value3", "column2": "value4", ...}}
]
```

2. For SERVICE TICKET and DETAILS OF DELIVERABLE tables, create THREE separate sections:

SECTION A - PROJECT INFORMATION (4 rows x 4 columns):
- Column headers: "Category", "Details", "Secondary_Info", "Date_Info"
- Row 1: {{"Category": "Well", "Details": "[Well Name]", "Secondary_Info": "Rig-up", "Date_Info": "[Rig-up Date]"}}
- Row 2: {{"Category": "Rig", "Details": "[Rig Name/Number]", "Secondary_Info": "Rig-down", "Date_Info": "[Rig-down Date]"}}
- Row 3: {{"Category": "Field", "Details": "[Field Name]", "Secondary_Info": "SO Number", "Date_Info": "