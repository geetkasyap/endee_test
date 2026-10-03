# Pull Request Description

**Title:** `fix(core,storage): resolve shutdown use-after-free, txn checks, and Windows portability`

### Summary
This PR addresses several stability, concurrency, and cross-platform issues across the core engine and storage layer:

1. **Clean Autosave Thread Teardown (`src/core/ndd.hpp`)**:
   - Replaced unconditional `std::this_thread::sleep_for` with `std::condition_variable` (`autosave_cv_`).
   - Removed `autosave_thread_.detach()` in `~IndexManager()`. The destructor now signals `autosave_cv_.notify_all()` and safely joins the thread, preventing heap-use-after-free when indices are destroyed during shutdown.

2. **Transaction Error Handling in IDMapper (`src/storage/id_mapper.hpp`)**:
   - Checked return values for `mdbx_txn_begin` and `mdbx_txn_commit` in `deletePoints()` and `getDeletedIds()`.
   - Prevents undefined behavior and null/garbage pointer dereferencing when transactions cannot be allocated.

3. **Cross-Platform Portability for Windows (`src/core/rebuild.cpp`)**:
   - Replaced POSIX-only `gmtime_r` in `Rebuild::timeToISO8601` with a cross-platform check supporting `gmtime_s` on Windows.

4. **Resilient Batch Deletion (`src/core/ndd.hpp`)**:
   - Added per-item exception handling in `deleteVectorsByIds()` to prevent an error on a missing or stale vector from terminating the remaining batch.
   - Added bounds check on `stored_ids` and restricted WAL deletion logging to successfully removed IDs.

### Testing
- Verified clean git diff with no whitespace errors.
- Verified thread synchronization flow during `IndexManager` destruction.
