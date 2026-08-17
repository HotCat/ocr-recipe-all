# PySide6 OpenFRP Vision Architecture

This architecture is based on the old `allInOne` application and the `ocr-recipe-demo` graph editor. The goal is not a direct port. The goal is to preserve the parts that made the old system reliable: an OpenGL frame loop, FRP-style event composition, typed runtime controls, async workers, camera/OCR/hardware isolation, and recipe persistence.

## What The Old System Is Really Doing

The old app has these important dependency roles:

- `Main.hs`: owns the GLFW/OpenGL/Dear ImGui frame loop, builds the Scheme environment, loads `fun.scm` and `my.scm`, calls Yampa `reactimate`, and pushes frame data to OpenGL textures.
- `Object.hs`: defines typed runtime controls (`CtrlItem`), keyboard input, stages, Scheme config templates, and external process lifecycle.
- `Animate.hs`: creates the mutable control map (`CtrlRef`) with sliders, checkboxes, ROIs, texture IDs, channels, OCR state, trigger state, and frame-time sampling.
- `ObjectBehavior.hs`: is the main FRP composition layer. `combineAll` wires UI panels, camera parameters, config select/save, ROI preview, barcode/kingpin workflow, Canny, serial OCR, counters, and production result state.
- `fun.scm` and `my.scm`: are runtime pipeline scripts. They decide which visual routines run before/after reading camera frames.
- `config*.scm`: are product recipe files. They store camera parameters, ROI rectangles, OpenGL preview controls, OCR scale factors, and counters.
- `TextRecog.hsc`, `Barcode.hsc`, `Sqlite.hs`: are side-effect boundaries. They run OCR/native OpenCV/SAM2/ZeroMQ/Modbus/calibration/SQLite workers via `TChan` or C++ FFI.

In the rewrite, Scheme should disappear. Its two jobs should become:

- A typed node graph for workflow order and production rules.
- A typed recipe document for parameters, ROIs, camera settings, node positions, and output routing.

## Core UI Shape

Use one central `CameraWorkbench` widget:

```text
QMainWindow
  CameraWorkbench
    QStackedLayout.StackAll
      CameraGLView       QOpenGLWidget, live frame texture
      NodeOverlayView    transparent QGraphicsView, node editor + ROI items
      HudOverlay         small floating controls and result badges
```

`CameraGLView` is always alive and always paints the latest frame. `NodeOverlayView` can be fully hidden, partly transparent, or mouse-transparent. The user controls it from `HudOverlay`, not from a detached editor page.

Recommended overlay modes:

- `Hidden`: graph invisible and mouse-transparent.
- `ROI`: only ROI rectangles and result annotations visible.
- `Graph`: nodes/edges visible over the frame.
- `Edit`: full node editing, selection, node library, inspector, and pan/zoom enabled.

The graph editor from `ocr-recipe-demo` is a good model for typed ports, DAG validation, and JSON save/load, but its `QGraphicsView` should become the overlay layer instead of the main page.

## Runtime Event Model

Replace Yampa with a small explicit FRP kernel:

```text
CameraThread -> FrameArrived(frame_id, timestamp, size)
Qt shortcuts -> ShortcutPressed(name)
Overlay      -> GraphEdited, RoiEdited, NodeSelected
Executor     -> WorkflowStarted, NodeFinished, WorkflowFinished, WorkflowFailed
Hardware     -> PlcReady, PlcAck, PlcError
Timer        -> Tick(dt)

events -> reducer(AppState, Event) -> AppState
events -> effect router -> camera/OCR/DB/Modbus/executor side effects
```

Keep state updates separate from side effects. This mirrors the old Yampa style where signal functions compute behavior and return IO callbacks, but in Python it is easier to audit and test.

## Async Snapshot Workflow

The live camera stream must never wait for OCR.

1. Camera adapter emits frames into a latest-frame buffer and a small ring buffer.
2. `CameraGLView` paints only the latest frame texture.
3. User presses a shortcut, for example `Ctrl+Return`.
4. Runtime copies the latest frame into an immutable `FrameSnapshot`.
5. Active recipe revision and snapshot are submitted to `WorkflowExecutor`.
6. Executor runs the typed DAG off the GUI thread.
7. Node outputs are returned as `NodeResult` objects with text, verdicts, previews, ROIs, timings, and errors.
8. GUI receives result signals and updates HUD, overlay annotations, production counters, SQLite, and hardware outputs.

The old `TChan` pattern maps to `queue.Queue`, `concurrent.futures`, and Qt signals. Heavy OCR/AI nodes should be process-isolated later, but the first experimental version can use `QThreadPool` or `ThreadPoolExecutor`.

## Package Layout

```text
src/openfrp_vision/
  app.py
  core/
    events.py          typed events, reducer, app state
    runtime.py         event bus and side-effect router
  camera/
    base.py            CameraAdapter, FrameSnapshot
    opencv_camera.py   first adapter for webcam/video file
    hikvision.py       later SDK adapter
    mindvision.py      later SDK adapter if still needed
  ui/
    main_window.py     QMainWindow and CameraWorkbench
    camera_preview.py  QOpenGLWidget live frame texture
    node_overlay.py    transparent graph overlay
    hud.py             floating toolbar, result badges, mode controls
    inspector.py       selected-node parameters
  workflow/
    model.py           typed DAG, recipe serialization
    executor.py        async snapshot DAG runner
    nodes.py           ROI, preprocess, OCR, regex, aggregate, output
  services/
    ocr.py             PaddleOCR/Tesseract/OpenAI/VLM adapters
    modbus.py          request/response hardware protocol
    production_db.py   SQLite results and approved recipe revisions
    calibration.py     OpenCV calibration and homography
```

## Node Types To Start With

Start with production OCR nodes that match the old app:

- `FrameInput`: exposes the captured snapshot.
- `ROI`: crops in image-pixel coordinates and stores normalized coordinates too.
- `CameraRectify`: undistort/perspective-correct from calibration.
- `Canny`: debug/lighting node like the old Canny panel.
- `Threshold`: fixed/Otsu/adaptive threshold.
- `OCR`: PaddleOCR/Tesseract node with language/device/scale/confidence parameters.
- `RegexCheck`: validates serial/lot rules.
- `SequenceCheck`: validates incrementing serial-number windows.
- `Aggregate`: ALL/ANY/custom decision.
- `SaveEvidence`: image/result persistence.
- `ModbusOutput`: output command preview first, real driver later.

Every node should declare input port types, output port type, default parameters, and whether it is realtime-safe. Execution should receive only immutable input values and params.

## Recipe Document

Use versioned JSON first. YAML is nice for humans, but JSON is easier for strict round trips from a node editor.

```json
{
  "format": "openfrp-vision/recipe/v1",
  "recipe_id": "g5-kingpin-front-back",
  "revision": 12,
  "camera": {
    "adapter": "opencv",
    "device": 0,
    "properties": {
      "exposure_us": 15000,
      "gamma": 14,
      "contrast": 147,
      "reverse_x": false,
      "reverse_y": true
    }
  },
  "overlay": {
    "visible": true,
    "mode": "roi"
  },
  "nodes": [],
  "edges": []
}
```

Do not store mutable production counts in the approved recipe. Store counters and inspection results in SQLite with recipe revision, timestamp, frame id, and evidence paths.

## Coordinate Rules

Use two coordinate spaces:

- `image`: original camera pixels.
- `overlay`: QGraphicsScene coordinates.

ROI nodes store both:

```json
{
  "x": 1141,
  "y": 187,
  "width": 115,
  "height": 594,
  "normalized": [0.8914, 0.1558, 0.0898, 0.495]
}
```

When camera resolution changes, normalized coordinates can rebuild pixel coordinates. During production execution, use pixel coordinates from the actual snapshot dimensions.

## Migration Mapping

| Old concept | New concept |
| --- | --- |
| Yampa `SF GameInput ...` | reducer + event streams + workflow executor |
| `CtrlItem` / `CtrlRef` | typed node params + app state |
| Scheme `fun.scm` | workflow graph execution order |
| Scheme `config*.scm` | versioned recipe JSON |
| Dear ImGui floating windows | transparent Qt overlay and HUD |
| OpenGL texture frame | `QOpenGLWidget` camera texture |
| `TChan` worker queues | executor queues + Qt signals |
| `procSerialNum`, `procBarcode` | OCR/template node processors |
| `modbusTrigger`, `modbusPos`, `modbusNeg` | hardware service with explicit request/ack states |
| `prod.db` | production event store with recipe revision |

## Development Phases

1. Build `CameraWorkbench` with synthetic/OpenCV camera frames and a hideable overlay.
2. Port the typed DAG model from the demo into `workflow/model.py`.
3. Add `FrameInput -> ROI -> Threshold -> OCR -> Regex -> Aggregate` execution on `Ctrl+Return`.
4. Add ROI editing directly on the preview and bind ROI items to node params.
5. Add SQLite evidence/result logging.
6. Add camera adapters for the real industrial camera.
7. Add Modbus request/response driver.
8. Add recipe approval/revision locking before production use.

The first milestone is successful if the camera preview stays smooth while pressing the shortcut repeatedly, old results remain visible as overlay annotations, and graph editing can be hidden without interrupting the live frame.
