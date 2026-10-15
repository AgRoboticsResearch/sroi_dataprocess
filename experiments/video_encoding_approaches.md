# Video Encoding Approaches During Recording

Comparison of how video encoding is handled across this repo and the two lerobot versions.

## Current: `record_realsense.py`

Frames are saved to disk as PNG/JPEG via `cv2.imwrite()` during recording. When `--encode-video` is set, encoding happens synchronously in `stop_episode()` — it reads all images back from disk, encodes to MP4, then deletes the originals.

Blocking: **yes**, next episode cannot start until encoding finishes.

With the new `--mem` flag (`--encode-video --mem`), frames are buffered as in-memory RGB arrays instead of being written to disk. `stop_episode()` encodes directly from memory, skipping the disk round-trip. Memory usage is printed every 30 frames. Behavior is unchanged when `--mem` is not set.

Blocking: **yes**, but faster than disk-based encoding.

## lerobot (older, `~/codes/lerobot`)

Images are written to disk via `AsyncImageWriter` during recording. `save_episode()` calls `_encode_video_worker()` which reads images back from disk and encodes to MP4. Uses `ProcessPoolExecutor(max_workers=num_cameras)` for parallel multi-camera encoding.

`batch_encoding_size` parameter (default 1):
- **1**: encode immediately after each episode (blocking)
- **N > 1**: defer encoding, accumulate N episodes as PNGs, encode all at once every Nth episode

Blocking: **yes** (default), or **every Nth episode** (batch mode).

Key files:
- `src/lerobot/datasets/lerobot_dataset.py` — `save_episode()`, `_encode_video_worker()`
- `src/lerobot/datasets/video_utils.py` — `encode_video_frames()`
- `src/lerobot/datasets/image_writer.py` — `AsyncImageWriter`

## lerobot-official (new, `~/codes/lerobot-official`)

Three encoding modes:

### 1. Default (same as older lerobot)
PNG → disk → read back → encode after each episode. Blocking.

### 2. Batch encoding (`batch_encoding_size > 1`)
Same as older lerobot — defer encoding to every Nth episode. Blocking at batch boundaries.

### 3. Streaming encoding (`streaming_encoding=True`) — NEW
Per-camera encoder threads (`StreamingVideoEncoder` → `_CameraEncoderThread`). Each camera gets a bounded queue (`queue_maxsize=30` by default). Frames are encoded to MP4 in real-time during recording via `feed_frame()`. PyAV's `encode()` releases the GIL, so encoding runs in parallel with capture.

- `save_episode()` is near-instant — just waits for encoder threads to flush
- If encoder queue is full, frames are **dropped with warnings** (non-blocking backpressure)
- No intermediate PNG files on disk

Blocking: **no**.

Key files:
- `src/lerobot/datasets/video_utils.py` — `StreamingVideoEncoder`, `_CameraEncoderThread`
- `src/lerobot/datasets/lerobot_dataset.py` — `add_frame()`, `save_episode()`, `finish_episode()`

## Summary

| Approach | Write to disk | Encoding timing | Blocks next episode | Risk |
|---|---|---|---|---|
| `record_realsense.py` default | PNG per frame | After episode | Yes | Slow disk I/O |
| `record_realsense.py --mem` | None | After episode (from RAM) | Yes (shorter) | Memory pressure on long episodes |
| lerobot default | PNG per frame | After episode | Yes | Slow disk I/O |
| lerobot batch | PNG per frame | Every N episodes | Every Nth episode | Disk space for N episodes |
| lerobot-official streaming | None | During recording | No | Frame drops if encoder can't keep up |

## Potential improvement for `record_realsense.py`

Adopt the streaming encoding approach from lerobot-official: spawn per-camera encoder threads during recording, feed frames via a bounded queue, and make `stop_episode()` a near-instant flush. This would eliminate the encoding pause between episodes entirely.

Considerations:
- IR images are grayscale (need `COLOR_GRAY2RGB` conversion) — do this before enqueuing
- Three streams (left, right, color) — one thread each
- Bounded queue size needs tuning to avoid frame drops at 30fps
- Memory usage is bounded by queue size × frame size × 3 streams (e.g. 30 × 640×480×3 × 3 ≈ 80 MB)
