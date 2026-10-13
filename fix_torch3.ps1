.\unsloth-env\Scripts\activate
pip uninstall -y torch torchvision torchaudio

$max_retries = 5;
$retry_count = 0;
while ($retry_count -lt $max_retries) {
    try {
        pip install --default-timeout=1000 torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124;
        if ($LASTEXITCODE -eq 0) { break; }
    } catch { }
    $retry_count++;
    Start-Sleep -Seconds 5;
}

python train.py
