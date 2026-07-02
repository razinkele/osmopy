---
name: Config key case sensitivity
description: Java OSMOSE is case-sensitive for config keys — writer must preserve original case from config files
type: feedback
---

Config reader lowercases keys for internal use, but Java OSMOSE is case-sensitive (e.g., `predation.predPrey.stage.threshold` ≠ `predation.predprey.stage.threshold`). The reader stores original case in `_last_key_case_map` (module-level in `osmose/config/reader.py`) and `write_temp_config` restores it when writing the merged config for Java.

**Why:** EEC config crashed with `ArrayIndexOutOfBoundsException` in `PredationMortality.getAccessibility` because Java couldn't find camelCase keys that were lowercased.

**How to apply:** When writing any config back to Java, always use `_last_key_case_map.get(key, key)` to restore original case. When adding new config keys programmatically (not from files), they won't be in the case map and will be written as-is.
