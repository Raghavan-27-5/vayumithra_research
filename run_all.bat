:: run_all.bat — Windows remote desktop: run full benchmark pipeline
:: Run from repo root: cd vayumithra_research && run_all.bat

@echo off
setlocal

echo ============================================================
echo VAYUMITHRA — Full Training Pipeline
echo ============================================================

:: Step 1 — Verify setup
echo.
echo [STEP 1] Setup check...
python scripts/setup_check.py
if %errorlevel% neq 0 (
    echo SETUP CHECK FAILED — fix errors above before continuing.
    pause
    exit /b 1
)

:: Step 2 — DLinear (engineered features)
echo.
echo [STEP 2] Training DLinear (engineered features)...
python scripts/train_dlinear.py --feature_mode engineered
if %errorlevel% neq 0 (echo DLinear training failed & pause & exit /b 1)

:: Step 3 — DLinear (raw sequences — for ablation comparison)
echo.
echo [STEP 3] Training DLinear (raw sequences)...
python scripts/train_dlinear.py --feature_mode raw --config configs/dlinear_config.yaml
if %errorlevel% neq 0 (echo DLinear-raw failed & pause & exit /b 1)

:: Step 4 — Mamba (engineered features)
echo.
echo [STEP 4] Training Mamba (engineered features)...
python scripts/train_mamba.py --feature_mode engineered
if %errorlevel% neq 0 (echo Mamba training failed & pause & exit /b 1)

:: Step 5 — Compare all models
echo.
echo [STEP 5] Generating benchmark table...
python scripts/compare_models.py
if %errorlevel% neq 0 (echo Benchmark comparison failed & pause & exit /b 1)

:: Step 6 — Commit results
echo.
echo [STEP 6] Committing results to git...
git add results/metrics/
git commit -m "results: full benchmark complete"
git push origin main

echo.
echo ============================================================
echo PIPELINE COMPLETE — check results/metrics/benchmark_table.csv
echo ============================================================
pause
