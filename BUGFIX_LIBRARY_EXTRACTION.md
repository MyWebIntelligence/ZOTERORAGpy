# Bugfix: Library Information Extraction

## Date
2025-10-26

## Problem
Error when generating Zotero notes: **"Error: Could not extract library information from any item URI"**

### Root Cause
The system was only trying to extract library information from the `uri` field in Zotero JSON exports. However:
1. Modern Zotero exports use the `library` field (with `id` and `type` subfields)
2. The `uri` field may not always be present in all export formats
3. The `itemKey` (or modern `key`) field was not being extracted from the JSON and stored in the CSV

## Solution

### 1. Multi-Method Library Info Extraction
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

## Files Modified

1. **app/utils/zotero_parser.py** (lines 81-222)
   - Added 3-method library extraction strategy
   - Added modern/legacy itemKey support
   - Updated documentation

2. **scripts/rad_dataframe.py** (lines 405-438)
   - Added `itemKey` field extraction
   - Added `abstract` field extraction
   - Added dual JSON format support

3. **app/templates/index.html** (lines 293-295, 865-866)
   - Changed model selection from dropdown to text input (separate feature)

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

**Before Fix:**
```
Error: Could not extract library information from any item URI
```

**After Fix:**
```
✅ Created: 5
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
