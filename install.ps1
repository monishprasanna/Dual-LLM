$ErrorActionPreference = "Continue"
$env:PIP_DEFAULT_TIMEOUT = "1000"

Write-Host "Activating environment..."
.\unsloth-env\Scripts\activate

$packages = @(
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "unsloth",
    "trl peft accelerate bitsandbytes",
    "gguf sentencepiece protobuf"
)

foreach ($pkg in $packages) {
    $success = $false
    $retries = 0
    while (-not $success -and $retries -lt 5) {
        Write-Host "Installing $pkg (Attempt $($retries + 1))..."
        Invoke-Expression "pip install $pkg"
        if ($LASTEXITCODE -eq 0) {
            $success = $true
            Write-Host "Successfully installed $pkg"
        } else {
            $retries++
            Write-Host "Installation failed. Retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }
    if (-not $success) {
        Write-Host "Failed to install $pkg after 5 attempts."
        exit 1
    }
}

Write-Host "All packages installed! Starting training..."
python train.py
