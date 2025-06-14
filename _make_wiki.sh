set -e
pages=(
  Home.md
  _Sidebar.md
  00‑Roadmap.md
  Phase‑A‑Intake.md
  Phase‑B‑Cleaning.md
  Phase‑C‑Dataset.md
  Phase‑D‑Fine‑tuning.md
  Phase‑E‑Evaluation.md
  Phase‑F‑Deployment.md
  Phase‑G‑Incremental‑Updates.md
  Data‑Privacy.md
)
for p in "${pages[@]}"; do
  [[ -f "$p" ]] || printf "# %s\n\n_Draft page stub._\n" "${p%.md}" > "$p"
done

# Create top‑level files if they don't exist
for p in "${pages[@]}"; do
  [[ -f "$p" ]] || printf "# %s\n\n_Draft page stub._\n" "${p%.md}" > "$p"
done

# Templates folder + sub‑pages
mkdir -p Templates
for t in Task‑Checklist.md Meeting‑Notes.md; do
  [[ -f "Templates/$t" ]] || printf "# Template – %s\n\n_Draft stub._\n" "${t%.md}" > "Templates/$t"
done
