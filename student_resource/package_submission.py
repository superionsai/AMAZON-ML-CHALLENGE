#!/usr/bin/env python3
"""Submission packaging script for ML Challenge 2026.

Creates a submission-ready zip archive following the exact competition structure:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import os
import sys
import zipfile
import subprocess

def package(team_name="TCC", out_dir="output", code_dir="code/business_entity_resolution",
            doc_file="Documentation_template.md", zip_name=None):
    if zip_name is None:
        zip_name = f"{team_name}_submission_v6.zip"

    # Step 1: Validate files exist
    m_tsv = os.path.join(out_dir, "matching_results.tsv")
    c_tsv = os.path.join(out_dir, "candidate_pairs.tsv")
    if not os.path.isfile(m_tsv):
        print(f"ERROR: {m_tsv} does not exist! Run predict first.")
        sys.exit(1)
    if not os.path.isfile(c_tsv):
        print(f"ERROR: {c_tsv} does not exist! Run predict first.")
        sys.exit(1)

    print("Step 1: Running official validator...")
    cmd = [sys.executable, "utils/validate_submission.py", "--matching", m_tsv, "--candidate", c_tsv, "--test-dir", "dataset/test"]
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("ERROR: Validation failed! Fix issues before packaging.")
        sys.exit(1)
    print("Validation passed successfully!\n")

    # Step 2: Create zip
    os.makedirs("../submissions", exist_ok=True)
    zip_path = os.path.join("../submissions", zip_name)
    print(f"Step 2: Creating archive: {zip_path}")
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        zf.write(m_tsv, arcname="output/matching_results.tsv")
        zf.write(c_tsv, arcname="output/candidate_pairs.tsv")
        print(f"  Added output/matching_results.tsv ({os.path.getsize(m_tsv):,} bytes)")
        print(f"  Added output/candidate_pairs.tsv ({os.path.getsize(c_tsv):,} bytes)")

        # Add code files
        for root, _, files in os.walk(code_dir):
            for f in files:
                if f.endswith((".py", ".txt", ".md", ".ps1")) and not f.startswith("."):
                    fp = os.path.join(root, f)
                    rel = os.path.relpath(fp, ".")
                    zf.write(fp, arcname=rel)
                    print(f"  Added {rel}")

        # Add documentation
        if os.path.isfile(doc_file):
            zf.write(doc_file, arcname="Documentation_template.md")
            print(f"  Added Documentation_template.md")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nSUCCESS! Submission archive created at:")
    print(f"  {os.path.abspath(zip_path)} ({zip_size_mb:.2f} MB)")
    print("Files in archive:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename:45s} {info.file_size:10,d} bytes (compressed: {info.compress_size:10,d})")

if __name__ == "__main__":
    package()
