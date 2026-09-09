# CogWAM Deployment

This deployment is isolated from the Pi05/OpenPI and FastWAM entry points.
Start it with `scripts/run_cogwam_server.sh` (WebSocket `127.0.0.1:7085`) and
use a separate tunnel to that port, then run
`scripts/run_cogwam_puzzle_client.sh 4` on the robot. The server always
predicts 64 waypoints; the client argument controls how many are executed per
request. The argument is the real client execution length and must be a
multiple of four, up to 64.

The server loads CogWAM from `/bh/zbh_self/projects/CogWAM` and its WAM
checkpoint/config. It requires Wan memory `first_layer_only`, with the trained
memory cadence: short-term offset `4`, current-subtask stride `4`, completed
subtask stride `16`, and five VLM memory frames. The WAM model predicts 64
joint waypoints; the bundled client executes the requested prefix in four-action
microchunks. After every microchunk it captures and sends an observation-only
update while a background reader collects the server response, so execution does
not wait for Wan/VLM processing. Wan memory and Q2 still see independently
observed frames at action steps `0, 4, 8, ...`. Thus `4`, `48`, and `64` are all
valid client horizons.

If Q2 confirms a subtask during a chunk, the server sets `stop_current_chunk`;
the client stops at the next four-action boundary and the next normal request
uses the updated subtask prompt. The server process and loaded models remain
running throughout.

Before returning actions, the CogWAM server clips all finite absolute joint
outputs to the configured Piper limits and applies the configured per-waypoint
`max_delta` from the measured current state. The 14-value order is
`[L1..L6,Lg,R1..R6,Rg]`; arm limits are radians and gripper limits are metres.

The configured VLM loads the Qwen backbone from
`/bh/zbh_ckp/models/Qwen3-VL-2B-Instruct` and the wrapper/tokenizer/query
embeddings from the supplied checkpoint directory, with both Q1 and Q2 LoRA
adapters. Q1 creates the initial plan once per episode. Q2 checks each non-final
subtask every four real action steps and requires three consecutive positive
checks before switching. Q1 runs when the next subtask starts, including the
initial episode plan; it is never a fixed four-step timer.
The final subtask is never passed to Q2. A higher-level controller may
send `final_task_done: true` in a request, or configure `final_done_file`; the
server then returns `final_task_done: true` for subsequent responses. The
controller should stop requesting new chunks after that response and perform
the normal hold/recording shutdown. This signal is intentionally external
because the Piper client has no RobotWin `task_env.success` interface.
The server rejects larger client horizons while Wan memory is enabled instead
of silently fabricating the missing intermediate frames.
