@echo off

echo [1/3] Running Preprocessing...
python preprocess_cwq.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [2/3] Extracting Embeddings...
python -m trm_agent.run --dataset cwq --stage embed --override embed_batch_size=256
if %errorlevel% neq 0 exit /b %errorlevel%

echo.
echo [3/3] Running Training...
call run_train_cwq.bat
