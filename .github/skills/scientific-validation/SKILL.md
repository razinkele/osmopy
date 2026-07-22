---
name: scientific-validation
description: >-
  Validate scientific claims, citations, and references against real published literature using the
  scite MCP server. Fire whenever the user is writing, editing, or reviewing scientific or technical
  prose that references published work — including docstrings, README sections, help modals, plan
  specs, commit messages, and research notes that cite authors, years, DOIs, journals, or
  "well-known" findings. Also fire on explicit asks to "validate", "verify", "check", "fact-check",
  or "look up" a claim, citation, DOI, author, or paper; to find supporting literature; to confirm
  a paper is real, not retracted, and actually says what it is being used to support. Skip only
  purely conversational mentions with no citation attached and edits that do not touch
  citation-bearing text.
license: MIT
compatibility: 'Cross-platform. Requires the scite MCP server to be configured and available.'
metadata:
  version: "1.0"
  categories:
    - Scientific writing
    - Citation validation
    - Literature review
    - Research integrity
argument-hint: 'Optional: paste a claim, citation, DOI, or text excerpt to validate'
---

# Scientific Validation

Validates scientific claims, citations, and references against the published literature using
the **scite MCP server** (`scite-search_literature` tool). Catches fabricated citations,
retracted papers, misattributed findings, and unsupported claims before they ship.

## Output Contract (Required)

Before finishing, produce a **Validation Report** containing:

1. **Citation inventory** — every claim/citation extracted from the input, numbered.
2. **Verification status** for each item (✅ Confirmed, ⚠️ Uncertain, ❌ Failed, 🔄 Retracted).
3. **Evidence** — DOI, title, authors, year, and Smart Citation tally for each verified paper.
4. **Accuracy assessment** — whether the claim text accurately represents what the cited paper says.
5. **Recommendations** — corrections, alternative citations, or flagged risks.

## Workflow

Copy and track this checklist:

```
- [ ] Phase 1: Extract claims and citations from input
- [ ] Phase 2: Resolve and verify each reference
- [ ] Phase 3: Check for retractions, corrections, and concerns
- [ ] Phase 4: Validate claim accuracy against paper content
- [ ] Phase 5: Produce the validation report
```

### Phase 1: Extract Claims and Citations

1. **Parse the input text** for anything that looks like a scientific reference:
   - Explicit DOIs (`10.xxxx/...`)
   - Author–year citations (`Smith et al., 2020`)
   - Paper titles in quotes or italics
   - Journal names with volume/page
   - Factual claims attributed to literature ("X has been shown to...", "According to Y...")
   - Unnamed but specific claims ("studies show that...", "it is well established that...")

2. **Build a citation inventory table:**

   | # | Claim / Citation Text | Type | Extracted Identifiers |
   |---|----------------------|------|----------------------|
   | 1 | "Smith et al. (2020) showed X" | Author-year | author: Smith, year: 2020 |
   | 2 | "doi:10.1038/s41586-020-2012-7" | DOI | DOI: 10.1038/... |
   | 3 | "Fish mortality follows a Type II functional response" | Factual claim | keywords: functional response, Type II, mortality |

### Phase 2: Resolve and Verify Each Reference

For each citation in the inventory, use the `scite-search_literature` tool to find the paper:

1. **DOI-based lookup** (most reliable — always prefer when a DOI is available):
   ```
   scite-search_literature({ dois: ["10.1038/s41586-020-2012-7"] })
   ```

2. **Title-based lookup** (when DOI is unavailable):
   ```
   scite-search_literature({ titles: ["Exact or near-exact paper title"] })
   ```

3. **Author + keyword search** (when only author-year is given):
   ```
   scite-search_literature({ term: "Smith functional response fish", author: "Smith", year: 2020 })
   ```

4. **Topical search** (for unnamed claims needing supporting literature):
   ```
   scite-search_literature({ term: "\"Type II functional response\" AND fish AND mortality", date_from: "2015" })
   ```

**For each resolved paper, record:**
- DOI, title, authors, year, journal
- `tally` — supporting, contrasting, and mentioning citation counts
- `isOa`, `oaStatus` — open access availability
- Whether the paper was actually found (vs. no results)

### Phase 3: Check for Retractions, Corrections, and Concerns

For every resolved paper, check the `retraction_notices` field in scite results. Additionally:

1. **Explicit retraction check:**
   ```
   scite-search_literature({ dois: ["10.xxxx/..."], has_retraction: true })
   ```

2. **Check for editorial concerns:**
   ```
   scite-search_literature({ dois: ["10.xxxx/..."], has_concern: true })
   ```

3. **Check for corrections/errata:**
   ```
   scite-search_literature({ dois: ["10.xxxx/..."], has_correction: true })
   ```

4. **Review the Smart Citation balance.** A paper with many contrasting citations may be
   controversial or disputed:

   | Tally Pattern | Interpretation |
   |--------------|----------------|
   | High supporting, low contrasting | Well-accepted finding |
   | Balanced supporting/contrasting | Debated or nuanced topic |
   | High contrasting, low supporting | Disputed — flag for review |
   | Very few citations total | Insufficient evidence of community acceptance |

### Phase 4: Validate Claim Accuracy

For each claim that references a specific finding, verify that the paper actually supports
what is being claimed:

1. **Read the abstract** from scite results to check alignment with the claim.

2. **Search for full-text excerpts** that confirm or deny the claim:
   ```
   scite-search_literature({ dois: ["10.xxxx/..."], term: "specific claim keywords" })
   ```

3. **Check Smart Citations** — these are actual sentences from other papers citing this work,
   classified as supporting, contrasting, or mentioning:
   - `citations[].snippet` — the exact text from a citing paper
   - `citations[].type` — "supporting", "contrasting", or "mentioning"
   - `citations[].section` — where in the citing paper this appears

4. **Classify the claim accuracy:**

   | Status | Meaning |
   |--------|---------|
   | ✅ Accurate | Paper exists, is not retracted, and the claim faithfully represents its findings |
   | ⚠️ Imprecise | Paper exists but the claim oversimplifies, exaggerates, or slightly misrepresents |
   | ⚠️ Unverifiable | Paper exists but full text is unavailable to confirm the specific claim |
   | ❌ Inaccurate | Paper exists but does not support the stated claim |
   | ❌ Not found | No matching paper could be located — possible fabrication |
   | 🔄 Retracted | Paper has been retracted — citation must be removed or flagged |
   | 🔄 Corrected | Paper has a published correction that may affect the cited finding |

### Phase 5: Produce the Validation Report

Format the final report as follows:

```markdown
## Scientific Validation Report

### Summary
- **Total citations checked:** N
- **Confirmed:** N ✅
- **Uncertain:** N ⚠️
- **Failed:** N ❌
- **Retracted/Corrected:** N 🔄

### Detailed Results

#### 1. [Claim or citation text]
- **Status:** ✅ Confirmed
- **Paper:** Author et al. (Year). "Title." *Journal*, Vol(Issue), Pages.
- **DOI:** 10.xxxx/...
- **Citation tally:** X supporting, Y contrasting, Z mentioning
- **Assessment:** The claim accurately reflects the paper's findings on...
- **Open Access:** Yes/No (link if available)

#### 2. [Claim or citation text]
- **Status:** ❌ Not found
- **Search performed:** [describe what was searched]
- **Recommendation:** Remove citation or replace with [suggested alternative]

### Recommendations
- [Actionable items: fix citations, add DOIs, replace retracted refs, etc.]
```

---

## Special Scenarios

### Finding Supporting Literature for Unattributed Claims

When the text makes a scientific claim without a citation ("studies show that..."):

1. Search scite for the claim's key concepts:
   ```
   scite-search_literature({ term: "relevant technical terms", date_from: "2010", supporting_from: 5 })
   ```
2. Prioritize papers with high supporting citation counts as authoritative sources.
3. Suggest specific papers the author could cite.

### Validating OSMOSE-Specific Scientific References

This project (OSMOSE marine ecosystem model) frequently references:
- Fish bioenergetics and growth models (von Bertalanffy)
- Predator-prey dynamics (Holling Type II/III functional responses)
- Size-spectrum ecology
- Individual-based / agent-based modeling
- Marine ecosystem trophic interactions

For OSMOSE-specific claims, also search for:
```
scite-search_literature({ term: "OSMOSE model marine ecosystem", date_from: "2000" })
```

Key OSMOSE papers to know:
- Shin & Cury (2001, 2004) — original OSMOSE model papers
- Travers et al. (2009) — OSMOSE-LTL coupling
- Grüss et al. — OSMOSE applications in the Gulf of Mexico

### Batch Validation of a Document

When validating an entire document or README with multiple citations:

1. Extract ALL citations first (Phase 1) before querying any.
2. Deduplicate — the same paper may be cited multiple times.
3. Batch DOI lookups where possible (pass multiple DOIs in one call).
4. Report results in document order for easy cross-referencing.

---

## Anti-Patterns

| ❌ Don't | ✅ Do instead |
|---------|--------------|
| Assume a citation is correct because it "looks right" | Always verify against scite |
| Report a paper as "not found" after a single vague search | Try DOI, title, author+year, and keyword searches |
| Ignore retraction notices | Always check `retraction_notices` and `has_retraction` |
| Validate only the paper's existence | Also verify the *claim* matches the paper's actual findings |
| Skip papers behind paywalls | Use Smart Citations (snippets from citing papers) as proxy evidence |
| Fabricate DOIs or paper details | Only report what scite actually returns |
| Silently pass claims with no citation | Flag unattributed scientific claims and suggest citations |

---

## Quick Reference: scite-search_literature Parameters

| Parameter | Use Case |
|-----------|----------|
| `dois` | Look up specific papers by DOI (most reliable) |
| `titles` | Look up papers by title (when DOI unavailable) |
| `term` | Full-text search with Boolean operators (AND, OR, NOT) |
| `author` | Filter by author name |
| `year` | Filter by publication year |
| `date_from` / `date_to` | Date range filter |
| `journal` | Filter by journal name |
| `has_retraction` | Find retracted papers |
| `has_concern` | Find papers with editorial concerns |
| `has_correction` | Find papers with corrections |
| `has_erratum` | Find papers with errata |
| `supporting_from` | Minimum supporting citations (find well-supported papers) |
| `contrasting_from` | Minimum contrasting citations (find debated papers) |

### scite Response Fields

| Field | Contains |
|-------|----------|
| `doi`, `title`, `authors`, `year`, `journal` | Paper metadata |
| `abstract` | Full abstract text |
| `tally.supporting/contrasting/mentioning` | Smart Citation counts |
| `fulltextExcerpts` | Matching passages from the paper (OA only) |
| `citations[].snippet` | Actual text from citing papers |
| `citations[].type` | "supporting", "contrasting", or "mentioning" |
| `retraction_notices` | Any retraction, correction, or concern notices |
| `access.url` | Best available link to the paper |
| `isOa`, `oaStatus`, `license` | Open access information |
