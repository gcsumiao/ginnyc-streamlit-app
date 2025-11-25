# Repository Guidelines

## Project Structure & Module Organization
- `DMM-1.ipynb`: Main notebook containing data prep and Streamlit dashboard code. Treat the cell with `# dmm_dashboard.py` as the app source; copy to `dmm_dashboard.py` if you want a standalone script.
- `US_AMAZON_blackBoxProducts_*.csv`: Helium10 exports; keep raw files unedited. Add new drops alongside existing ones.
- `outputs/`: Place generated summaries (e.g., Excel reports). Keep outputs derived from current data and code.

## Build, Test, and Development Commands
- Run dashboard (if exported): `python3 -m streamlit run dmm_dashboard.py` from repo root. Ensure CSV paths point to local files.
- Notebook exploration: Open `DMM-1.ipynb` in Jupyter/VS Code; rerun cells to refresh analyses.
- Dependency install: `python3 -m pip install streamlit pandas numpy`. Pin versions if sharing results.

## Coding Style & Naming Conventions
- Language: Python (pandas + Streamlit). Use 4-space indents, snake_case for variables/functions, uppercase for constants.
- Data paths: Prefer `Path(__file__).parent / "data" / "file.csv"` over absolute paths. Keep file names descriptive and date-stamped (YYYY-MM-DD).
- Caching: Use `@st.cache_data` with a version flag or file hash to avoid stale dashboards after data updates.

## Testing Guidelines
- No automated tests yet; validate manually: run dashboard, verify filters, and sanity-check KPIs (revenues/sales totals) against the CSVs.
- When adding functions, include quick `assert` checks or small sample dataframes in notebooks/scripts to confirm calculations.

## Commit & Pull Request Guidelines
- Commits: Write clear, imperative messages (e.g., "Add price tier summary to dashboard"). Group related changes; avoid committing raw data unless necessary.
- PRs: Describe the change, data sources used, and verification steps (dashboard run, KPI spot-check). Attach screenshots/gifs of the dashboard when UI changes. Link any tracking issue or task.

## Security & Configuration Tips
- Keep raw Helium10 exports private; do not publish. Remove credentials/API keys from notebooks before sharing.
- When sharing the app, switch to relative paths or environment variables for file locations to avoid leaking local directories.
