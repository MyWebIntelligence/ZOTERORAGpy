# Bugfix: Library Information Extraction

## Date
2025-10-26

## Updates
- **Initial fix:** 2025-10-26 - Multi-method extraction strategy
- **Additional fix #1:** 2025-10-26 - Added `load_dotenv()` to `zotero_parser.py` to enable .env fallback
- **Additional fix #2:** 2025-10-26 - Added `load_dotenv()` to `llm_note_generator.py` to enable LLM clients + default model support
- **Additional fix #3:** 2025-10-26 - Added `safe_str()` function to handle pandas NaN/float values in metadata

## Problems

### Problem 1: Library Information Extraction
Error when generating Zotero notes: **"Error: Could not extract library information. Please set ZOTERO_USER_ID or ZOTERO_GROUP_ID in Settings."**

### Problem 2: LLM Not Used
Warning in logs: **"LLM not available or disabled, using template"** - Notes generated with placeholder text instead of intelligent LLM-generated content

### Problem 3: Type Error in Prompt Building
Error in logs: **"LLM generation failed, using template fallback: replace() argument 2 must be str, not float"** - Prompt generation crashes when metadata contains pandas NaN values

### Root Causes

**Issue 1:** Limited extraction methods
- The system was only trying to extract library information from the `uri` field in Zotero JSON exports
- Modern Zotero exports use the `library` field (with `id` and `type` subfields)
- The `uri` field may not always be present in all export formats
- The `itemKey` (or modern `key`) field was not being extracted from the JSON and stored in the CSV

**Issue 2:** Missing .env loading in zotero_parser.py
- The fallback method (Method 3) reads `ZOTERO_USER_ID` and `ZOTERO_GROUP_ID` from environment variables
- However, `zotero_parser.py` was calling `os.getenv()` without first loading the `.env` file with `load_dotenv()`
- This caused the fallback to always fail even when credentials were properly configured in `.env`

**Issue 3:** Missing .env loading in llm_note_generator.py
- The `llm_note_generator.py` module was calling `os.getenv("OPENAI_API_KEY")` and `os.getenv("OPENROUTER_API_KEY")` without loading `.env` first
- This caused the LLM clients (`openai_client` and `openrouter_client`) to never be initialized
- Result: The condition `if use_llm and (openai_client or openrouter_client):` always evaluated to False, forcing fallback to template
- The default model was also hardcoded to `"gpt-4o-mini"` instead of using `OPENROUTER_DEFAULT_MODEL` from `.env`

**Issue 4:** Type error with pandas NaN values
- When reading CSV files, pandas can return `float('nan')` for empty cells
- The `_build_prompt()` function was doing `template.replace("{DATE}", date)` where `date` could be a `float` NaN
- Python's `str.replace()` requires the second argument to be a string, not a float
- This caused `TypeError: replace() argument 2 must be str, not float` and forced fallback to template

## Solution

### 1. Added .env Loading
**File:** `app/utils/zotero_parser.py` (lines 14-17)

Added the missing `load_dotenv()` call at module initialization:

```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
```

This ensures that the fallback method (Method 3) can properly read `ZOTERO_USER_ID` and `ZOTERO_GROUP_ID` from the `.env` file.

### 2. Multi-Method Library Info Extraction
**File:** `app/utils/zotero_parser.py`

Modified `extract_library_info_from_session()` to try 3 extraction methods in order:

1. **Method 1 (Modern format):** Extract from `item["library"]["id"]` and `item["library"]["type"]`
   ```python
   library_field = item.get("library")
   if library_field and isinstance(library_field, dict):
       lib_type = library_field.get("type")  # "user" or "group"
       lib_id = library_field.get("id")
   ```

2. **Method 2 (Legacy format):** Extract from `item["uri"]` using regex pattern
   ```python
   uri = item.get("uri")
   lib_info = extract_library_info_from_uri(uri)  # Regex: zotero.org/(users|groups)/(\d+)/items/...
   ```

3. **Method 3 (Fallback):** Use credentials from `.env` file
   ```python
   ZOTERO_USER_ID or ZOTERO_GROUP_ID
   ```

### 2. ItemKey Extraction Support
**File:** `app/utils/zotero_parser.py`

Modified `extract_item_keys_from_json()` to support both modern and legacy field names:

```python
# Modern exports use "key", legacy use "itemKey"
item_key = item.get("itemKey") or item.get("key")
```

### 3. CSV Data Extraction Enhancement
**File:** `scripts/rad_dataframe.py`

Added missing fields to metadata extraction:

```python
metadata = {
    "itemKey": item.get("key") or item.get("itemKey", ""),  # NEW
    "type": item.get("itemType", ""),
    "title": item.get("title", ""),
    "abstract": item.get("abstractNote", ""),  # NEW
    "date": item.get("date", ""),
    "url": item.get("url", ""),
    "doi": item.get("DOI", ""),
    "authors": "...",
}
```

Also added support for both Zotero JSON formats:
- Format 1: Direct array `[{item1}, {item2}, ...]`
- Format 2: Object with items key `{"items": [{item1}, {item2}, ...]}`

### 4. Existing CSV Compatibility
**File:** `app/main.py`

The code already supports multiple column name variations:
- Library info: Uses `.env` fallback if JSON extraction fails
- ItemKey: Searches for `"itemKey"`, `"item_key"`, or `"key"` columns (line 712)
- Abstract: Searches for `"abstractNote"` or `"abstract"` columns (line 752)

### 5. LLM Client Initialization Fix
**File:** `app/utils/llm_note_generator.py` (lines 14-17, 27)

Added `load_dotenv()` to properly initialize LLM clients:

```python
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ✅ Now loads from .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # ✅ Now loads from .env
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "gpt-4o-mini")  # NEW
```

Result: Both `openai_client` and `openrouter_client` are now properly initialized when keys are present in `.env`.

### 6. Default Model Support
**Files:** `app/utils/llm_note_generator.py`, `app/templates/index.html`, `app/main.py`

Implemented support for `OPENROUTER_DEFAULT_MODEL` from `.env`:

**Backend (`llm_note_generator.py`):**
```python
def _generate_with_llm(prompt: str, model: str = None, ...):
    # Use OPENROUTER_DEFAULT_MODEL if no model specified
    if not model:
        model = OPENROUTER_DEFAULT_MODEL
        logger.info(f"No model specified, using default: {model}")
```

**Frontend (`index.html` line 866):**
```javascript
const model = modelInput || ''; // Send empty string to use server default
```

**Endpoint (`main.py` line 621):**
```python
model: str = Form("")  # Empty string → uses OPENROUTER_DEFAULT_MODEL
```

**Result:** When user leaves model field empty, system uses `google/gemini-2.5-flash` (or whatever is configured in `.env`) via OpenRouter.

### 7. Type-Safe Metadata Conversion
**File:** `app/utils/llm_note_generator.py` (lines 105-114, 275-284)

Added `safe_str()` helper function to convert all metadata values to strings:

```python
def safe_str(value, default="N/A"):
    """Convert value to string, handling NaN and None."""
    if value is None or value == "":
        return default
    # Check for pandas NaN (float type)
    if isinstance(value, float):
        import math
        if math.isnan(value):
            return default
    return str(value)

# Usage in _build_prompt()
title = safe_str(metadata.get("title"), "Sans titre")
authors = safe_str(metadata.get("authors"), "N/A")
date = safe_str(metadata.get("date"), "N/A")
doi = safe_str(metadata.get("doi"), "")
url = safe_str(metadata.get("url"), "")
```

**Result:** All metadata values are safely converted to strings before being used in `template.replace()`, preventing `TypeError` with pandas NaN values.

## Files Modified

1. **app/utils/zotero_parser.py** (lines 14-17, 81-222, 270-288)
   - **CRITICAL:** Added `load_dotenv()` to enable .env fallback
   - Added 3-method library extraction strategy
   - Added modern/legacy itemKey support
   - Updated documentation

2. **app/utils/llm_note_generator.py** (lines 14-17, 27, 105-136, 184-207, 275-291, 307-349)
   - **CRITICAL:** Added `load_dotenv()` to initialize LLM clients
   - **CRITICAL:** Added `safe_str()` helper function to handle pandas NaN values
   - Added `OPENROUTER_DEFAULT_MODEL` support
   - Modified `_generate_with_llm()` to accept `model=None` and use default
   - Modified `build_note_html()` to accept `model=None` and use default
   - Applied `safe_str()` to all metadata in `_build_prompt()` and `_fallback_template()`
   - Updated documentation

3. **scripts/rad_dataframe.py** (lines 405-438)
   - Added `itemKey` field extraction
   - Added `abstract` field extraction
   - Added dual JSON format support

4. **app/templates/index.html** (lines 293-295, 866)
   - Changed model selection from dropdown to text input
   - Changed default from `'gpt-4o-mini'` to `''` (empty string for server default)

5. **app/main.py** (lines 621, 635-636, 641, 772-778)
   - Changed endpoint default from `Form("gpt-4o-mini")` to `Form("")`
   - Added logic to convert empty string to None before calling `build_note_html()`
   - Updated documentation

## Testing

### Test Scenarios

1. ✅ **Modern Zotero export** (with `library` field)
   - Should extract library info from `item.library.id` and `item.library.type`
   - Should extract itemKey from `item.key`

2. ✅ **Legacy Zotero export** (with `uri` field only)
   - Should extract library info from URI regex
   - Should extract itemKey from `item.itemKey`

3. ✅ **Minimal export** (no library info in JSON)
   - Should fallback to `ZOTERO_USER_ID` or `ZOTERO_GROUP_ID` from Settings
   - User must configure credentials in Settings panel

4. ✅ **Both JSON formats**
   - Direct array: `[{...}, {...}]`
   - Object wrapper: `{"items": [{...}, {...}]}`

### Expected Behavior

#### Library Extraction (Problem 1)

**Before Fix:**
```
ERROR: Could not extract library information from any item URI
```

**After Fix:**
```
WARNING: Could not extract library info from JSON items, checking .env credentials
INFO: Using ZOTERO_USER_ID from .env: 15681
INFO: Detected library: type=users, id=15681
```

#### LLM Generation (Problem 2)

**Before Fix:**
```
INFO: LLM not available or disabled, using template
```
→ Notes contain placeholder text: "à compléter" / "to be completed"

**After Fix:**
```
INFO: OpenRouter client initialized for note generation
INFO: No model specified, using default: google/gemini-2.5-flash
INFO: Using OpenRouter with model: google/gemini-2.5-flash
INFO: Generated note content (length: 1543 chars)
```
→ Notes contain intelligent LLM-generated analysis

#### Type Error (Problem 3)

**Before Fix:**
```
ERROR: LLM generation failed, using template fallback: replace() argument 2 must be str, not float
```
→ Prompt building crashes when CSV contains empty cells (pandas NaN)

**After Fix:**
```
INFO: Loaded prompt template from /Users/.../zotero_prompt.md
INFO: Using OpenRouter with model: google/gemini-2.5-flash
INFO: Generated note content (length: 1543 chars)
```
→ All metadata safely converted to strings before prompt building

#### Final Result
```
✅ Created: 5 notes with LLM-generated content
ℹ️ Already exists: 0
⏭️ Skipped: 0
❌ Errors: 0
```

## Migration Notes

### For Existing Users

If you already processed a Zotero export **before this fix**, you need to:

1. **Re-process your ZIP file** from Step 2 (Extract Text & Metadata) to regenerate `output.csv` with the new `itemKey` and `abstract` columns
2. **Or manually add credentials** in Settings:
   - Open Settings (⚙️ icon)
   - Add your `ZOTERO_USER_ID` (find it at https://www.zotero.org/settings/keys)
   - Save

### For New Users

No action needed - the fix is automatic for all new uploads.

## Error Message Changes

**Old error message:**
```
Error: Could not extract library information from any item URI
```

**New error message (if all 3 methods fail):**
```
Error: Could not extract library information. Please set ZOTERO_USER_ID or ZOTERO_GROUP_ID in Settings.
```

## Related Documentation

- Original feature implementation: `CHANGELOG_PROMPT_CUSTOMIZATION.md`
- API compliance validation: `ZOTERO_API_COMPLIANCE.md`
- JSON format fix: `BUGFIX_ZOTERO_JSON_FORMAT.md`
- **This fix**: `BUGFIX_LIBRARY_EXTRACTION.md`
