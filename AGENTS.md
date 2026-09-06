# Project Rules & Constraints

These global rules apply to all tasks and operations in this workspace:

1. **Package Management**: Never reinstall a package that is already in `requirements.txt` or was already imported successfully earlier in the session.
2. **Chunked & Incremental Processing**: Never load an entire large file into memory at once. Always process in chunks (e.g., `pandas` with `chunksize`, or streaming reads) and write intermediate results to disk incrementally.
3. **Execution Transparency**: Before running any script, print only a short summary of what it will do and estimated memory/row count. Do not run exploratory system commands (e.g., `ps aux`, `free -m`) or full file dumps unless an actual failure occurs.
4. **Failure Handling**: If a script fails, state the exact error message and the root-cause fix immediately. Do not blindly re-run the same failing approach.
