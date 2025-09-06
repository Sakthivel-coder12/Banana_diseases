@echo off
setlocal enabledelayedexpansion

:: Paths (edit these if needed)
set "orig_dir=dataset_custom\train\disease"
set "proc_dir=preprocessed_custom_fixed_1\train\disease"

:: Counter
set count=1

:: Loop through all files in original directory
for %%f in ("%orig_dir%\original_*.*") do (
    set "num=0000!count!"
    set "num=!num:~-4!"

    :: Rename processed file with same index
    if exist "%proc_dir%\processed_!num!.png" (
        ren "%proc_dir%\processed_!num!.png" "processed_!num!.png"
    )

    set /a count+=1
)

echo Renaming complete.
pause
