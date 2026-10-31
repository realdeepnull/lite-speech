| runtime  | device | dtype | phase | download MB | load first ms | TTFA cold ms | TTFA p50 ms | TTFA p95 ms | RTF p50 | RTF p95 | peak RSS MB | cancel p50 ms | runs |
| -------- | ------ | ----- | ----- | ----------- | ------------- | ------------ | ----------- | ----------- | ------- | ------- | ----------- | ------------- | ---- |
| chromium | wasm   | q4    | cold  | 68.0        | 6719          | 100          | 74          | 100         | 0.417   | 0.481   | —           | 0.3           | 5    |
| chromium | wasm   | q4    | warm  | 0.0         | 1075          | 88           | 75          | 88          | 0.413   | 0.441   | —           | 0.2           | 5    |
| chromium | webgpu | q4    | cold  | 68.0        | 6863          | 680          | 29          | 680         | 0.316   | 0.413   | —           | 0.2           | 5    |
| chromium | webgpu | q4    | warm  | 0.0         | 1226          | 87           | 27          | 87          | 0.336   | 0.343   | —           | 0.2           | 5    |
| chromium | webgpu | q4f16 | cold  | 41.3        | 5856          | 618          | 33          | 618         | 0.317   | 0.404   | —           | 0.2           | 5    |
| chromium | webgpu | q4f16 | warm  | 0.0         | 1117          | 97           | 31          | 97          | 0.322   | 0.349   | —           | 0.2           | 5    |
| node     | cpu    | fp32  | cold  | 161.9       | 8196          | 58           | 44          | 58          | 0.258   | 0.275   | 403.6       | 0.5           | 5    |
| node     | cpu    | fp32  | warm  | 0.3         | 1127          | 106          | 46          | 106         | 0.302   | 0.361   | 515.0       | 2.1           | 5    |
| node     | cpu    | q4    | cold  | 68.0        | 5405          | 110          | 38          | 110         | 0.216   | 0.244   | 304.3       | 0.5           | 5    |
| node     | cpu    | q4    | warm  | 0.3         | 1181          | 91           | 46          | 91          | 0.241   | 0.310   | 356.9       | 0.4           | 5    |
| node     | cpu    | q4f16 | cold  | 41.3        | 5073          | 68           | 46          | 68          | 0.296   | 0.304   | 307.7       | 0.6           | 5    |
| node     | cpu    | q4f16 | warm  | 0.3         | 1281          | 27           | 47          | 50          | 0.317   | 0.335   | 312.4       | 0.5           | 5    |

TTFA = time to first audio chunk. RTF = stream wall time / audio duration (lower is faster).
"load first ms" is the first `preload()` in the process (n = 1), because later streams reuse it.
Times are milliseconds from the engine's own `time-to-first-chunk` event, not from a timer
around the iterator. Peak RSS is the whole process, not a browser tab budget.
