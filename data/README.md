# `data/` - Contents and How to Populate It

## `corpus/` - RAG source documents

**Only `.txt` and `.md` files are read** by
`src/retrieval.py::build_parent_child_index`. Any other format (PDF, DOCX,
HTML...) is silently ignored - the retriever will simply act as if that
file does not exist. If you add a source in another format, **convert it
to `.md`/`.txt` first** (e.g. with `pdftotext -layout file.pdf file.md`)
and place the converted file here.

Current files:

| File | Content |
|---|---|
| `receiving_city_capacity.md` | Housing vacancy thresholds, the 3 public-service capacity proxies, capacity-weighting logic |
| `push_pull_factors.md` | Social networks as a pull factor, economic push factors, climate as a push factor |
| `migration_corridors.md` | Definition of a migration corridor, early warning indicators |
| `climate_early_warning_and_policy.md` | Data sources for climate migration early warning systems, proactive policy responses |
| `l-avenir-des-villes-face-aux-migrations-climatiques-2020-1.pdf`, `Migration_and_Cities_An_Introduction.pdf`, `Mateo Merchan Article City Migration.pdf` | Source/reference PDFs kept for provenance. **Not read by the retriever** (wrong format) - none of them have been converted to `.md`/`.txt`, so none of their content is currently searchable. If you want their content in the RAG index, convert them first (see the `pdftotext` command above) and add a row here. |

If you change topic or add new corpus documents, keep the same rule:
plain `.md`/`.txt`, one topic per file, and update the table above.

## `cities/cities.json`

Structured per-city data consumed by `get_city_capacity_profile` and
`compute_push_pull_index` in `src/mcp_server.py`. Each city needs:
`name`, `country`, `population`, `housing_vacancy_rate`,
`job_growth_rate_yoy`, `industry_diversity_index`,
`school_capacity_utilization`, `healthcare_beds_per_1000`,
`public_transit_coverage_index`. Add a new city by appending an object
with the same keys to the `"cities"` list.

## `eval_questions.json`

Questions + `ground_truth` reference answers used by `eval/ragas_eval.py`
and `eval/benchmark.py`. The rubric requires **≥10 questions** for full
marks (F, 11-12 pts) - there are currently 12. Each `ground_truth` should
be answerable from the content in `corpus/` - if you add a question, make
sure the supporting facts actually exist in a corpus document, or RAGAS's
`context_recall`/`faithfulness` scores will look artificially bad (not
because retrieval is broken, but because the answer was never retrievable
in the first place).