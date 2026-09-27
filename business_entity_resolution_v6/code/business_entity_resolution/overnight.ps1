# Overnight chain for V6. Run from student_resource\ :
#   powershell -ExecutionPolicy Bypass -File code\business_entity_resolution\overnight.ps1
# Each step logs to logs\*.txt. A failed step never stops the next one.
$py = (Get-Command python).Source
if (-not (Test-Path $py)) {
    $py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
}
$run = "code\business_entity_resolution\src\run.py"
$val = "utils\validate_submission.py"
New-Item -ItemType Directory -Force logs | Out-Null
function Step($name, [string[]]$argv) {
    "==== $(Get-Date -Format HH:mm:ss) START $name" | Tee-Object -Append logs\overview.txt
    & $py @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "logs\$name.txt"
    "==== $(Get-Date -Format HH:mm:ss) END   $name (exit $LASTEXITCODE)" | Tee-Object -Append logs\overview.txt
}
# A) V6 High-Recall + Safety Net + Singleton Gate (Primary Run)
Step "A1_train_v6"   @($run, "train", "--train-s1-frac", "0.4", "--model-dir", "models_v6")
Step "A2_predict_v6" @($run, "predict", "--model-dir", "models_v6", "--out-dir", "output_v6", "--n-threads", "8")
Step "A3_valid_v6"   @($val, "--matching", "output_v6\matching_results.tsv", "--candidate", "output_v6\candidate_pairs.tsv", "--test-dir", "dataset\test")

# B) V6 with Neural Bi-Encoder + Cross-Encoder (if CUDA is used)
# Step "B1_train_nn"   @($run, "train", "--train-s1-frac", "0.25", "--use-emb", "true", "--use-ce", "true", "--model-dir", "models_nn")
# Step "B2_predict_nn" @($run, "predict", "--model-dir", "models_nn", "--out-dir", "output_nn", "--n-threads", "6")
# Step "B3_valid_nn"   @($val, "--matching", "output_nn\matching_results.tsv", "--candidate", "output_nn\candidate_pairs.tsv", "--test-dir", "dataset\test")

"==== ALL DONE $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append logs\overview.txt
