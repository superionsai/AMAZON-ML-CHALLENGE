# Overnight chain. Run from student_resource\ :
#   powershell -ExecutionPolicy Bypass -File code\business_entity_resolution\overnight.ps1
# Each step logs to logs\*.txt. A failed step never stops the next one.
$py  = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
$run = "code\business_entity_resolution\src\run.py"
$val = "utils\validate_submission.py"
New-Item -ItemType Directory -Force logs | Out-Null
function Step($name, [string[]]$argv) {
    "==== $(Get-Date -Format HH:mm:ss) START $name" | Tee-Object -Append logs\overview.txt
    & $py @argv 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "logs\$name.txt"
    "==== $(Get-Date -Format HH:mm:ss) END   $name (exit $LASTEXITCODE)" | Tee-Object -Append logs\overview.txt
}
# A) full model: density-correct training + multilingual embeddings + cross-encoder
Step "A1_train_nn"   @($run, "train", "--train-s1-frac", "0.2", "--use-emb", "true", "--use-ce", "true", "--model-dir", "models_nn")
Step "A2_predict_nn" @($run, "predict", "--model-dir", "models_nn", "--out-dir", "output_nn", "--n-threads", "6")
Step "A3_valid_nn"   @($val, "--matching", "output_nn\matching_results.tsv", "--candidate", "output_nn\candidate_pairs.tsv", "--test-dir", "dataset\test")
# B) safety net: same density fix, no neural parts
Step "B1_train_v3"   @($run, "train", "--train-s1-frac", "0.2", "--model-dir", "models_v3")
Step "B2_predict_v3" @($run, "predict", "--model-dir", "models_v3", "--out-dir", "output_v3", "--n-threads", "6")
Step "B3_valid_v3"   @($val, "--matching", "output_v3\matching_results.tsv", "--candidate", "output_v3\candidate_pairs.tsv", "--test-dir", "dataset\test")
"==== ALL DONE $(Get-Date -Format HH:mm:ss)" | Tee-Object -Append logs\overview.txt
