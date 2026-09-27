#!/usr/bin/env python3
"""
TCC - Submission Packager
Run from student_resource/ directory.

Usage:
  python3 scripts/package_submission.py <cv_score> <candidate_density> "changelog notes"

Example:
  python3 scripts/package_submission.py 0.7823 12.4 "V0 TF-IDF char ngram baseline, threshold=0.55"
"""
import os
import sys
import glob
import json
import shutil
import zipfile
import subprocess
from datetime import datetime

SUBMISSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "submissions")
SUBMISSIONS_DIR = os.path.normpath(SUBMISSIONS_DIR)
TRACKER_FILE    = os.path.join(SUBMISSIONS_DIR, "SUBMISSION_TRACKER.md")
TEAM_NAME       = "TCC"
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR        = os.path.dirname(SCRIPT_DIR)   # student_resource/


def get_next_submission_id() -> str:
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(SUBMISSIONS_DIR, "submission_*"))
    nums = []
    for p in existing:
        part = os.path.basename(p).split("_")
        if len(part) == 2 and part[1].isdigit():
            nums.append(int(part[1]))
    return f"submission_{max(nums, default=0) + 1}"


def package(sub_id: str, cv_score: float, density: float, notes: str):
    sub_dir = os.path.join(SUBMISSIONS_DIR, sub_id)
    os.makedirs(sub_dir, exist_ok=True)

    matching_tsv  = os.path.join(BASE_DIR, "output", "matching_results.tsv")
    candidate_tsv = os.path.join(BASE_DIR, "output", "candidate_pairs.tsv")
    doc_template  = os.path.join(BASE_DIR, "Documentation_template.md")

    # ── 1. Validate ────────────────────────────────────────────────────────────
    print(f"\n[*] Step 1/5: Validating {sub_id} against official rules...")
    res = subprocess.run([
        sys.executable,
        os.path.join(BASE_DIR, "utils", "validate_submission.py"),
        "--matching",   matching_tsv,
        "--candidate",  candidate_tsv,
        "--test-dir",   os.path.join(BASE_DIR, "dataset", "test")
    ])
    if res.returncode != 0:
        print("[!] Validation FAILED. Fix errors then re-run.")
        sys.exit(1)
    print("    -> PASS")

    # ── 2. Copy TSVs to versioned directory ────────────────────────────────────
    print(f"[*] Step 2/5: Copying output files to {sub_dir}/")
    dest_matching  = os.path.join(sub_dir, "matching_results.tsv")
    dest_candidate = os.path.join(sub_dir, "candidate_pairs.tsv")
    shutil.copy2(matching_tsv, dest_matching)
    shutil.copy2(candidate_tsv, dest_candidate)

    # ── 3. Build TCC_submission.zip with EXACT official internal structure ─────
    zip_filename = f"{TEAM_NAME}_submission.zip"
    zip_path     = os.path.join(sub_dir, zip_filename)
    print(f"[*] Step 3/5: Building {zip_filename} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # Required: output/matching_results.tsv
        z.write(dest_matching, "output/matching_results.tsv")
        # Required: output/candidate_pairs.tsv
        z.write(dest_candidate, "output/candidate_pairs.tsv")
        # Required: Documentation_template.md (at root)
        if os.path.isfile(doc_template):
            z.write(doc_template, "Documentation_template.md")
        else:
            print(f"    [WARN] {doc_template} not found – skipping from zip")

        # Required: code/business_entity_resolution/ tree
        code_root = os.path.join(BASE_DIR, "code", "business_entity_resolution")
        if os.path.isdir(code_root):
            for root, dirs, files in os.walk(code_root):
                # Exclude artifacts that must NOT be in the zip
                dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", ".git", "node_modules")]
                for fname in files:
                    if fname.endswith((".pyc", ".pyo", ".DS_Store", ".gitignore")):
                        continue
                    full = os.path.join(root, fname)
                    arc  = os.path.relpath(full, BASE_DIR)  # relative to student_resource/
                    z.write(full, arc)
        else:
            print(f"    [WARN] Code dir {code_root} not found – skipping from zip")

    print(f"    -> {zip_path}")

    # ── 4. Write local CHANGELOG.md (NOT inside zip) ───────────────────────────
    changelog = os.path.join(sub_dir, "CHANGELOG.md")
    print(f"[*] Step 4/5: Writing CHANGELOG.md ...")
    with open(changelog, "w", encoding="utf-8") as f:
        f.write(f"# Changelog: {sub_id}\n\n")
        f.write(f"- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Local CV Macro F_0.5**: `{cv_score:.4f}`\n")
        f.write(f"- **Avg candidates/entity**: `{density:.2f}`\n\n")
        f.write(f"## Changes\n{notes}\n")

    # ── 5. Save metadata.json ──────────────────────────────────────────────────
    meta = {
        "submission_id": sub_id,
        "team_name": TEAM_NAME,
        "timestamp": datetime.now().isoformat(),
        "local_macro_f05": cv_score,
        "avg_candidate_density": density,
        "zip_path": zip_path,
        "notes": notes.splitlines()[0][:120],
    }
    with open(os.path.join(sub_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[*] Step 5/5: Updating SUBMISSION_TRACKER.md ...")

    # ── 6. Append to master tracker ────────────────────────────────────────────
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    if not os.path.isfile(TRACKER_FILE):
        with open(TRACKER_FILE, "w", encoding="utf-8") as f:
            f.write("# TCC – Submission Performance Tracker\n\n")
            f.write("| Submission | Date | Local F_0.5 | Public LB | Avg Candidates | Notes |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
    with open(TRACKER_FILE, "a", encoding="utf-8") as f:
        first_line = notes.splitlines()[0][:80] if notes.strip() else "-"
        f.write(
            f"| [{sub_id}](./{sub_id}/) "
            f"| {datetime.now().strftime('%Y-%m-%d %H:%M')} "
            f"| {cv_score:.4f} "
            f"| Pending "
            f"| {density:.1f} "
            f"| {first_line} |\n"
        )

    print(f"\n[+] Done! Packaged as: {zip_path}")
    print(f"    Upload to leaderboard: {dest_matching}")
    print(f"    Final zip (if required): {zip_path}")


if __name__ == "__main__":
    sub_id = get_next_submission_id()
    cv     = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    dens   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    notes  = sys.argv[3]        if len(sys.argv) > 3 else "No notes provided"
    package(sub_id, cv, dens, notes)
