"""
Neon Grid Drone Simulation — Comprehensive User Manual PDF Generator
=====================================================================
Generates 'drone_operations_manual.pdf' with complete instructions,
edge-case handling, debugging guide, and state-machine behaviour reference.
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ================================================================
#  Styles
# ================================================================

_STYLES = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "Title", parent=_STYLES["Heading1"],
    fontName="Helvetica-Bold", fontSize=24, leading=30,
    textColor=colors.black, alignment=1, spaceAfter=6,
)
SUBTITLE = ParagraphStyle(
    "Subtitle", parent=_STYLES["Normal"],
    fontName="Helvetica", fontSize=11, leading=15,
    textColor=colors.dimgrey, alignment=1, spaceAfter=30,
)
H1 = ParagraphStyle(
    "H1", parent=_STYLES["Heading2"],
    fontName="Helvetica-Bold", fontSize=15, leading=19,
    textColor=colors.black, spaceBefore=16, spaceAfter=8,
    keepWithNext=True,
)
H2 = ParagraphStyle(
    "H2", parent=_STYLES["Heading3"],
    fontName="Helvetica-Bold", fontSize=12, leading=16,
    textColor=colors.black, spaceBefore=10, spaceAfter=6,
    keepWithNext=True,
)
BODY = ParagraphStyle(
    "Body", parent=_STYLES["BodyText"],
    fontName="Helvetica", fontSize=9.5, leading=13.5,
    textColor=colors.black, spaceAfter=6,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY,
    leftIndent=18, bulletIndent=6, spaceAfter=4,
)
NOTE = ParagraphStyle(
    "Note", parent=BODY,
    fontName="Helvetica-Oblique", fontSize=9, leading=12,
    textColor=colors.black,
    leftIndent=12, borderPadding=4, spaceAfter=8,
)
CODE = ParagraphStyle(
    "Code", parent=_STYLES["Normal"],
    fontName="Courier", fontSize=8.5, leading=11,
    textColor=colors.black,
    leftIndent=12, spaceAfter=6,
)

TABLE_BODY = ParagraphStyle(
    "TableBody", parent=BODY,
    fontSize=8.5, leading=11, spaceAfter=0,
)

def make_table(data, colWidths):
    # Wrap data cells in Paragraph to enable text wrapping
    for i in range(1, len(data)):
        for j in range(len(data[i])):
            if isinstance(data[i][j], str):
                data[i][j] = Paragraph(data[i][j], TABLE_BODY)
    t = Table(data, colWidths=colWidths)
    t.setStyle(_table_style())
    return t

# Table styling helper
def _table_style():
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.black),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.white),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.whitesmoke]),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
    ])

HR = HRFlowable(width="100%", thickness=0.5, color=colors.grey,
                 spaceBefore=6, spaceAfter=6)


# ================================================================
#  Content Builder
# ================================================================

def build_manual():
    pdf_path = "drone_operations_manual.pdf"
    doc = SimpleDocTemplate(
        pdf_path, pagesize=letter,
        rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42,
    )
    S = []  # story

    # ── Cover ──────────────────────────────────────────────────
    S.append(Spacer(1, 60))
    S.append(Paragraph("Drone Simulation", TITLE))
    S.append(Paragraph("User Manual", SUBTITLE))
    S.append(Paragraph(
        "Autonomous Navigation and Obstacle Avoidance Simulation",
        ParagraphStyle("sub2", parent=SUBTITLE, fontSize=9.5, spaceAfter=40),
    ))
    S.append(HR)
    S.append(Spacer(1, 10))

    # ── Table of Contents ──────────────────────────────────────
    S.append(Paragraph("Table of Contents", H1))
    toc_items = [
        "1.  System Requirements &amp; Installation",
        "2.  Project File Structure",
        "3.  Launching the Simulation",
        "4.  Environment &amp; Coordinate System",
        "5.  Keyboard Controls Reference (Complete)",
        "6.  Standard Flight Operations (Step-by-Step)",
        "7.  Terrain Scanning System",
        "8.  Obstacle Avoidance",
        "9.  Behavior Priorities",
        "10. Path Selection (A* / Dijkstra)",
        "11. Sensor Filtering",
        "12. Output Files",
        "13. Error Handling",
        "14. Troubleshooting Guide",
        "15. Appendix",
    ]
    for t in toc_items:
        S.append(Paragraph(t, BULLET))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  1. System Requirements
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("1. System Requirements &amp; Installation", H1))
    S.append(Paragraph(
        "The simulation requires a Windows system with Python 3.10 or above. "
        "The following packages are necessary:", BODY))
    deps = [
        "<b>• mujoco</b> (≥ 3.0): Physics engine and viewer.",
        "<b>• numpy</b>: Numerical array processing.",
        "<b>• Pillow (PIL)</b>: Image generation for terrain maps.",
        "<b>• matplotlib</b>: Sensor filtering comparison graphs.",
        "<b>• reportlab</b>: PDF generation for reports and this manual.",
    ]
    for d in deps:
        S.append(Paragraph(d, BULLET))
    S.append(Spacer(1, 4))
    S.append(Paragraph(
        "Install all dependencies via: <font face='Courier'>"
        "pip install mujoco numpy pillow matplotlib reportlab</font>", BODY))
    S.append(Paragraph(
        "<b>Note:</b> The <font face='Courier'>run.bat</font> launcher script automatically "
        "checks for and installs missing <font face='Courier'>mujoco</font> and "
        "<font face='Courier'>pillow</font> packages. Other packages must be installed manually.", NOTE))
    S.append(HR)

    # ════════════════════════════════════════════════════════════
    #  2. Project File Structure
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("2. Project File Structure", H1))
    file_data = [
        ["File", "Type", "Purpose"],
        ["scene.xml", "MuJoCo Model", "Complete world definition: terrain, pads, rocks, drone, dynamic obstacles."],
        ["simulate.py", "Main Entry", "Simulation loop, autopilot, key handling, viewer rendering, all integrations."],
        ["sensor_simulation.py", "Sensor Suite", "Virtual sensors (GPS, IMU, Baro, Lidar, Front), Kalman/EMA/Comp filters."],
        ["terrain_scanner.py", "Feature 1", "Onboard terrain scanning with directional/omni sensor and Fog-of-War grid."],
        ["dynamic_obstacles.py", "Feature 2", "Runtime obstacle spawning, path blocking, and occupancy grid updates."],
        ["subsumption.py", "Feature 3", "Behavior priority manager: Kill Switch, Avoidance, Navigation, Scanning."],
        ["generate_assets.py", "Asset Gen", "Generates neon grid texture and rough terrain heightmap PNG."],
        ["add_rocks.py", "Asset Gen", "Procedurally generates 50 stone obstacles into scene.xml."],
        ["generate_pdf.py", "Report Gen", "Generates the detailed project report PDF with benchmarks."],
        ["generate_manual_pdf.py", "Manual Gen", "Generates this user manual PDF."],
        ["run.bat", "Launcher", "One-click launcher: checks deps, generates assets, runs simulation."],
    ]
    t = make_table(file_data, colWidths=[120, 70, 330])
    S.append(t)

    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Generated output files:</b>", BODY))
    out_files = [
        "<b>• ground_map.png</b>: 2D occupancy map with Fog-of-War overlay showing explored / unexplored regions.",
        "<b>• shortest_path_map.png</b>: Same as above but with the planned path drawn on top.",
        "<b>• map_data.json</b>: Full serialised occupancy grid, landing spots, path waypoints, and metadata.",
        "<b>• occupancy_grid.csv</b>: Raw 160×160 grid (0=free, 1=obstacle, -1=unknown).",
        "<b>• flight_trajectory.csv</b>: Timestamped position, quaternion, velocity, and forces log.",
        "<b>• detection_log.csv</b>: Flight events (takeoff, scan, pad arrival, landing, kill switch).",
        "<b>• sensor_filtering_analysis.png</b>: 6-panel comparative analysis graph saved on simulation exit.",
    ]
    for o in out_files:
        S.append(Paragraph(o, BULLET))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  3. Launching the Simulation
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("3. Launching the Simulation", H1))
    S.append(Paragraph(
        "<b>Method 1 — run.bat (Recommended):</b> Double-click <font face='Courier'>run.bat</font>. "
        "It runs <font face='Courier'>generate_assets.py</font> (grid texture + heightmap), then "
        "<font face='Courier'>add_rocks.py</font> (50 stone obstacles into scene.xml), "
        "then launches <font face='Courier'>simulate.py</font>.", BODY))
    S.append(Paragraph(
        "<b>Method 2 — Manual:</b> Open a terminal in the project directory and run:", BODY))
    S.append(Paragraph("python generate_assets.py", CODE))
    S.append(Paragraph("python add_rocks.py", CODE))
    S.append(Paragraph("python simulate.py", CODE))
    S.append(Paragraph(
        "<b>Important:</b> You must click inside the MuJoCo viewer window after it opens "
        "to capture keyboard input. Without clicking, keypresses are not forwarded.", NOTE))
    S.append(HR)

    # ════════════════════════════════════════════════════════════
    #  4. Environment & Coordinate System
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("4. Environment &amp; Coordinate System", H1))
    S.append(Paragraph(
        "The arena spans <b>16 m × 16 m</b> (X from −8.0 to +8.0, Y from −8.0 to +8.0). "
        "The terrain is a rough Martian/Lunar heightfield with flattened circles at each landing pad. "
        "Six landing pads are pre-configured:", BODY))
    pad_data = [
        ["Pad Name", "World X", "World Y", "Radius (m)", "Key Binding"],
        ["Home", "−4.0", "−4.0", "0.6", "H"],
        ["Pad 1", "−4.0", " 4.0", "0.4", "1"],
        ["Pad 2", " 4.0", " 4.0", "0.4", "2"],
        ["Pad 3", " 4.0", "−4.0", "0.4", "3"],
        ["Pad 4", " 0.0", " 2.0", "0.4", "4"],
        ["Pad 5", " 0.0", "−2.0", "0.4", "5"],
    ]
    t = make_table(pad_data, colWidths=[80, 60, 60, 70, 80])
    S.append(t)
    S.append(Paragraph(
        "The occupancy grid maps this 16 m² world into a <b>160 × 160</b> integer grid. "
        "Each cell is approximately <b>0.1 m × 0.1 m</b>. Cells with value 1 are obstacles; "
        "value 0 are free; value −1 are unexplored (Fog of War).", BODY))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  5. Keyboard Controls Reference (Complete)
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("5. Keyboard Controls Reference (Complete)", H1))
    S.append(Paragraph(
        "All keys are case-insensitive. You must click inside the viewer window first.", BODY))
    ctrl_data = [
        ["Key", "Function", "Required State", "Detailed Behaviour"],
        ["L", "Takeoff / Land", "Any",
         "If LANDED: resets PID integrators, enters TAKEOFF, climbs to 1.5 m hover height. "
         "If airborne (HOVER/SCANNING/GOTO_TARGET/AUTOLAND): enters LANDING_HOME, descends in-place."],
        ["C", "Terrain Scan", "HOVER only",
         "Initiates a 360° yaw spin. During spin, the onboard scanner reveals cells within sensor radius. "
         "After full rotation, transitions to HOVER and marks the occupancy grid as ready."],
        ["S", "Toggle Path Overlay", "Any airborne",
         "Shows/hides the green path capsule line. If path shown and target is set: starts GOTO_TARGET flight. "
         "If path hidden during TAKEOFF or GOTO_TARGET: immediately transitions to HOVER."],
        ["1–5", "Select Target Pad", "HOVER / LANDED / GOTO_TARGET",
         "Calculates path from current position to selected pad using the active algorithm (A* or Dijkstra). "
         "Path is computed but flight does NOT start until S is pressed."],
        ["H", "Select Home Pad", "HOVER / LANDED / GOTO_TARGET",
         "Same as 1–5 but targets the Home pad at (−4, −4)."],
        ["O", "Spawn Obstacle", "Any",
         "If GOTO_TARGET: spawns an obstacle 1.2–2.5 m ahead on the planned path. "
         "Otherwise: spawns at a random valid coordinate avoiding landing pads."],
        ["K", "Kill Switch", "Any",
         "Immediately cuts all motor thrust via the Priority 5 subsumption behaviour. "
         "Clears all path data, resets to LANDED. The drone will fall under gravity."],
        ["P", "Toggle Algorithm", "Any",
         "Switches the pathfinder between A* and Dijkstra. The change applies to the next path calculation."],
        ["R", "Toggle Random Spawning", "Any",
         "Enables/disables periodic random obstacle spawning every 12 seconds. "
         "If the drone is flying, obstacles are spawned in front of it on the path."],
    ]
    t = make_table(ctrl_data, colWidths=[30, 95, 125, 270])
    S.append(t)
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  6. Standard Flight Operations
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("6. Standard Flight Operations (Step-by-Step)", H1))

    S.append(Paragraph("6.1 Basic Flight to a Pad", H2))
    steps_basic = [
        "<b>Step 1 — Takeoff:</b> Press <b>L</b>. The drone climbs from the home pad to 1.5 m hover altitude. "
        "Console prints <font face='Courier'>[CTRL] Takeoff!</font> and <font face='Courier'>[CTRL] Hover reached</font>.",
        "<b>Step 2 — Scan (first flight only):</b> Press <b>C</b>. The drone performs a 360° spin, "
        "revealing terrain within a 2.5 m radius via its directional sensor. Console prints <font face='Courier'>[SCAN] 360° scan complete!</font>.",
        "<b>Step 3 — Select Target:</b> Press a pad key (e.g., <b>2</b> for Pad 2). Console prints "
        "<font face='Courier'>[NAV] Planning A* from ... → ...</font> and shows waypoint count / cost.",
        "<b>Step 4 — Start Flight:</b> Press <b>S</b>. The green path line appears and the drone begins following it. "
        "The behaviour manager activates <font face='Courier'>Navigation (Priority 2)</font>.",
        "<b>Step 5 — Arrival:</b> When the drone reaches within 30 cm of the target pad, it automatically transitions to "
        "AUTOLAND → DESCENDING → WAITING (2 s) → LANDED. Path data is cleared.",
    ]
    for s in steps_basic:
        S.append(Paragraph(s, BULLET))
    S.append(Spacer(1, 6))

    S.append(Paragraph("6.2 Flight with Terrain Discovery", H2))
    S.append(Paragraph(
        "On the very first run, the map starts mostly unexplored (Fog of War). The scanner reveals cells "
        "as the drone moves. The planner uses an <b>optimistic strategy</b>: unexplored cells are treated "
        "as free space. If the drone later discovers an obstacle in an unexplored region while navigating, "
        "the subsumption avoidance layer detects the path blockage and automatically replans.", BODY))

    S.append(Paragraph("6.3 Flight with Dynamic Obstacle Avoidance", H2))
    steps_dyn = [
        "<b>Step 1:</b> Begin a flight to any pad (Steps 1–4 above).",
        "<b>Step 2 — Spawn Obstacle:</b> Press <b>O</b> while the drone is in GOTO_TARGET. An obstacle spawns "
        "1.2–2.5 m ahead on the planned path. Console prints <font face='Courier'>[DYNAMIC] Spawned ...</font>.",
        "<b>Step 3 — Automatic Replanning:</b> The scanner detects the obstacle at the blocked waypoint. The subsumption "
        "avoidance layer (Priority 3) pauses navigation, recalculates the path, rebuilds the waypoints, and resumes flight. "
        "Console prints <font face='Courier'>[AVOIDANCE] Re-routing ...</font>.",
        "<b>Step 4:</b> The drone seamlessly continues to the target on the new route.",
    ]
    for s in steps_dyn:
        S.append(Paragraph(s, BULLET))
    S.append(Spacer(1, 6))

    S.append(Paragraph("6.4 Using Periodic Random Obstacle Spawning", H2))
    S.append(Paragraph(
        "Press <b>R</b> to enable periodic spawning. Every 12 seconds, there is an 80% chance that a new "
        "obstacle is generated. If the drone is actively navigating, the obstacle appears directly on its path; "
        "otherwise it spawns at a random arena coordinate. Press <b>R</b> again to disable. "
        "Up to 5 dynamic obstacles can be active simultaneously; the oldest is recycled when all 5 are used.", BODY))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  7. Terrain Scanning & Fog-of-War
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("7. Terrain Scanning System", H1))
    S.append(Paragraph(
        "The terrain scanner (<font face='Courier'>terrain_scanner.py</font>) maintains an internal "
        "<b>discovered_grid</b> and <b>explored_mask</b>. When the simulation starts, only the immediate "
        "vicinity of all landing pads is pre-revealed. The rest of the 160×160 grid is marked as "
        "unknown (value −1).", BODY))
    S.append(Paragraph("<b>Sensor Modes:</b>", BODY))
    sensor_modes = [
        "<b>• Directional (Default):</b> A 90° forward-facing cone sensor with 2.5 m range. Cells must be "
        "within the FOV and range to be revealed. This models a front-facing depth camera or LiDAR.",
        "<b>• Omnidirectional:</b> Reveals all cells within the sensor radius regardless of heading. "
        "This models a 360° scanning LiDAR. Can be enabled programmatically.",
    ]
    for s in sensor_modes:
        S.append(Paragraph(s, BULLET))
    S.append(Paragraph(
        "The drone's yaw heading determines the scan direction. During a 360° spin scan (C key), the "
        "directional sensor sweeps the full circle, achieving near-omnidirectional coverage.", BODY))
    S.append(Paragraph(
        "<b>Optimistic Planning:</b> The <font face='Courier'>get_planning_grid()</font> method returns "
        "a grid where all unknown cells (−1) are mapped to free space (0). This means the pathfinder "
        "will plan routes through unexplored terrain. If the drone later discovers an obstacle there, "
        "the path is invalidated and replanned.", BODY))
    S.append(Paragraph(
        "<b>Visual Output:</b> The <font face='Courier'>ground_map.png</font> file uses a dark "
        "Fog-of-War overlay for unexplored cells, bright red for discovered obstacles, and the neon grid "
        "background for explored free space.", BODY))
    S.append(HR)

    # ════════════════════════════════════════════════════════════
    #  8. Dynamic Obstacle System
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("8. Obstacle Avoidance", H1))
    S.append(Paragraph(
        "Five static MuJoCo bodies (<font face='Courier'>dynamic_obstacle_1</font> through "
        "<font face='Courier'>dynamic_obstacle_5</font>) are defined in <font face='Courier'>scene.xml</font> "
        "at z = −10 m (underground, invisible). When spawned, they are repositioned to the arena surface "
        "using <font face='Courier'>model.body_pos</font> and become visible glowing magenta obstacles.", BODY))
    S.append(Paragraph("<b>Spawning Methods:</b>", BODY))
    spawn_methods = [
        "<b>• Keyboard (O):</b> Instant manual spawn. If navigating, spawns on path ahead; otherwise random.",
        "<b>• Periodic Random (R):</b> Automatic spawn every 12 seconds (80% probability).",
        "<b>• Programmatic:</b> Call <font face='Courier'>obstacle_manager.spawn_obstacle(model, data, x, y, grid)</font>.",
    ]
    for s in spawn_methods:
        S.append(Paragraph(s, BULLET))
    S.append(Paragraph(
        "<b>Obstacle Recycling:</b> When all 5 obstacle slots are used and a new spawn is requested, "
        "the oldest active obstacle is removed from the occupancy grid and repositioned to the new location "
        "(FIFO recycling). Landing pad zones are always protected from obstacle placement.", BODY))
    S.append(Paragraph(
        "<b>Grid Updates:</b> Each spawned obstacle marks occupied cells on the ground truth grid "
        "using a circular mask. The scanner must then discover these cells for the planner to see them. "
        "The viewer draws a translucent magenta ring around each active obstacle's base.", BODY))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  9. Subsumption Architecture
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("9. Behavior Priorities", H1))
    S.append(Paragraph(
        "The subsumption architecture wraps the existing autopilot controller with a priority-based "
        "decision layer. At each simulation step, the BehaviorManager evaluates all behaviours from "
        "highest to lowest priority. The first behaviour whose <font face='Courier'>check_trigger()</font> "
        "returns True gets exclusive control of the drone.", BODY))
    prio_data = [
        ["Priority", "Behaviour Name", "Trigger Condition", "Action"],
        ["5 (Highest)", "Kill Switch",
         "kill_switch_active == True",
         "Immediately clears all forces, sets LANDED, clears path data."],
        ["4", "Emergency Landing",
         "state == LANDING_HOME",
         "Descends in-place using PID controller, transitions to LANDED on ground contact."],
        ["3", "Obstacle Avoidance",
         "state == GOTO_TARGET AND path_blocked == True",
         "Pauses navigation, recalculates path from current position using active algorithm, resumes."],
        ["2", "Navigation",
         "state in {TAKEOFF, HOVER, GOTO_TARGET, AUTOLAND}",
         "Normal flight operations: climb, hover, path-follow, autoland at destination pad."],
        ["1 (Lowest)", "Terrain Scanning",
         "state == SCANNING",
         "Performs 360° yaw spin for terrain mapping, maintains altitude."],
    ]
    t = make_table(prio_data, colWidths=[55, 80, 140, 245])
    S.append(t)
    S.append(Paragraph(
        "<b>Behaviour Override Example:</b> If the drone is scanning (Priority 1) and the kill switch is pressed, "
        "the Kill Switch behaviour (Priority 5) takes over immediately, cutting all forces. "
        "Similarly, if a path blockage is detected during navigation (Priority 2), the Obstacle Avoidance "
        "behaviour (Priority 3) temporarily takes control to replan, then navigation resumes.", BODY))
    S.append(HR)

    # ════════════════════════════════════════════════════════════
    #  10. Pathfinder Selection
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("10. Path Selection (A* / Dijkstra)", H1))
    S.append(Paragraph(
        "Press <b>P</b> at any time to toggle between A* and Dijkstra. The currently selected algorithm "
        "is used for all subsequent path calculations (both user-commanded and automatic replanning). "
        "The console prints the active algorithm after toggling.", BODY))
    algo_data = [
        ["Property", "A* (Default)", "Dijkstra"],
        ["Heuristic", "Octile distance (admissible for 8-connected grid)", "None (uniform cost)"],
        ["Optimality", "Optimal (same cost as Dijkstra)", "Optimal"],
        ["Node Expansion", "Significantly fewer (directed search)", "Explores all reachable nodes"],
        ["Speed", "Faster (especially on large grids)", "Slower (exhaustive)"],
        ["Use Case", "Real-time point-to-point navigation", "Full-tour path computation"],
    ]
    t = make_table(algo_data, colWidths=[90, 210, 220])
    S.append(t)
    S.append(Paragraph(
        "Both algorithms use the same 8-connected grid movement with obstacle inflation cost: cells "
        "adjacent to obstacles incur a +3.0 cost penalty to maintain safe clearance.", BODY))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  11. Sensor Simulation & Filtering
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("11. Sensor Filtering", H1))
    S.append(Paragraph(
        "The sensor simulation suite (<font face='Courier'>sensor_simulation.py</font>) runs at every "
        "physics timestep (200 Hz). It reads ground truth from MuJoCo and injects realistic Gaussian noise.", BODY))
    sens_data = [
        ["Sensor", "Rate", "Noise Std Dev", "Frame", "Measurement"],
        ["GPS", "10 Hz", "XY: 0.05 m, Z: 0.10 m", "World", "Position (x, y, z)"],
        ["IMU Accelerometer", "200 Hz", "0.15 m/s²", "Body", "Proper acceleration (ax, ay, az)"],
        ["IMU Gyroscope", "200 Hz", "0.02 rad/s", "Body", "Angular velocity (ωx, ωy, ωz)"],
        ["Barometer", "200 Hz", "0.10 m", "World", "Absolute altitude (z)"],
        ["Altimeter (LiDAR)", "200 Hz", "0.015 m", "Drone-down", "Distance to ground"],
        ["Front Rangefinder", "200 Hz", "0.03 m", "Drone-forward", "Distance to nearest obstacle (mj_ray)"],
    ]
    t = make_table(sens_data, colWidths=[90, 45, 95, 55, 230])
    S.append(t)
    S.append(Spacer(1, 6))
    S.append(Paragraph("<b>Implemented Filters:</b>", BODY))
    filters = [
        "<b>• EMA (α=0.15):</b> Simple low-pass filter. Good noise reduction, but introduces phase delay.",
        "<b>• Complementary (α=0.98):</b> Fuses gyroscope (high-frequency, low drift) with accelerometer "
        "(low-frequency, accurate DC) for attitude estimation. Also fuses accelerometer with barometer for altitude.",
        "<b>• Kalman Filter (2-state):</b> Optimal kinematic estimator [position, velocity]. Uses acceleration "
        "as control input and GPS/barometer as measurements. Best overall RMSE and zero phase delay.",
    ]
    for f in filters:
        S.append(Paragraph(f, BULLET))
    S.append(Paragraph(
        "On simulation exit, a 6-panel comparison plot is saved to "
        "<font face='Courier'>sensor_filtering_analysis.png</font>.", BODY))
    S.append(HR)

    # ════════════════════════════════════════════════════════════
    #  12. Telemetry & Output Files
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("12. Output Files", H1))
    S.append(Paragraph(
        "<b>flight_trajectory.csv</b> logs every 20th physics step (~10 Hz effective rate): "
        "time, state, X, Y, Z, quaternion (qw, qx, qy, qz), velocities (vx, vy, vz), "
        "and applied forces (fx, fy, fz).", BODY))
    S.append(Paragraph(
        "<b>detection_log.csv</b> records discrete flight events: TAKEOFF, HOVER_REACHED, SCAN_START, "
        "SCAN_COMPLETE, GOTO_TARGET_START, PAD_ARRIVED, TOUCHDOWN, LANDED, EMERGENCY_LAND, "
        "REPLAN_SUCCESS, REPLAN_FAILED, KILL_SWITCH_TRIGGERED.", BODY))
    S.append(Paragraph(
        "<b>map_data.json</b> is a complete JSON dump of the occupancy grid, landing spot metadata, "
        "the current shortest path waypoints, and grid metadata (origin, resolution, dimensions).", BODY))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  13. Edge Cases & Unexpected Input Handling
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("13. Error Handling", H1))
    S.append(Paragraph(
        "This section documents how the simulation handles unusual user interactions and boundary conditions. "
        "Every scenario has been analyzed by studying the key callback, state machine, and behaviour manager.", BODY))

    edge_cases = [
        ("<b>Pressing L while LANDED:</b>",
         "Initiates normal takeoff. Safe."),
        ("<b>Pressing L while already airborne (HOVER / SCANNING / GOTO_TARGET / AUTOLAND):</b>",
         "Enters LANDING_HOME state. The Emergency Landing behaviour (Priority 4) activates and descends "
         "in-place. Path data is cleared. The drone lands at its current X/Y position."),
        ("<b>Pressing C while LANDED or during flight (not HOVER):</b>",
         "Ignored. Console prints: <font face='Courier'>[SCAN] Must be HOVER. Current: &lt;state&gt;</font>. "
         "No state change occurs."),
        ("<b>Pressing S with no path calculated:</b>",
         "Console prints: <font face='Courier'>[VIZ] No path yet.</font> No state change."),
        ("<b>Pressing S to hide path during GOTO_TARGET flight:</b>",
         "Immediately transitions to HOVER. The drone stops mid-flight and holds position."),
        ("<b>Pressing a pad key (1-5, H) while SCANNING:</b>",
         "Rejected. <font face='Courier'>cmd_goto()</font> requires HOVER, LANDED, or GOTO_TARGET. "
         "Console prints the requirement."),
        ("<b>Pressing a pad key while already navigating to that pad:</b>",
         "Ignored. Console prints: <font face='Courier'>[NAV] Already flying to &lt;pad&gt;.</font>"),
        ("<b>Pressing a pad key while navigating to a different pad:</b>",
         "Accepted. New path is calculated from the current drone position. Path replaced; old path discarded."),
        ("<b>Pressing O when no inactive obstacles remain:</b>",
         "The oldest active obstacle is recycled (FIFO). It is removed from the grid and repositioned."),
        ("<b>Pressing K at any time:</b>",
         "The Kill Switch behaviour (Priority 5) immediately fires. All motor thrust is zeroed. "
         "All path state is cleared. The drone enters LANDED (it may drop if airborne due to gravity). "
         "This is by design — it simulates an emergency motor cutoff."),
        ("<b>Pressing K while LANDED:</b>",
         "The kill_switch_active flag is set, but on the next step the LANDED fast-path in the "
         "behaviour manager clears forces before the Kill Switch can fire. Harmless — no state change."),
        ("<b>Pressing P during active flight:</b>",
         "Algorithm is toggled. The current flight continues on the existing path. The new algorithm "
         "is only used for the next planning call (next pad selection or replanning event)."),
        ("<b>Pressing R during LANDED:</b>",
         "Periodic spawning is enabled, but obstacles only spawn when the timer fires. "
         "If the drone is not flying, obstacles spawn at random arena positions."),
        ("<b>Multiple rapid key presses:</b>",
         "Each keypress is processed sequentially in the key_callback. The simulation loop runs at "
         "200 Hz, so key events are handled between physics steps. No race conditions."),
        ("<b>Path to a blocked/unreachable pad:</b>",
         "The pathfinder returns None. Console prints: <font face='Courier'>[NAV] A*/Dijkstra failed</font>. "
         "No state change. User may scan more terrain or remove obstacles."),
        ("<b>Obstacle spawned exactly on a landing pad:</b>",
         "Protected by pad-avoidance logic. The spawn_random() and _update_grid_obstacle() methods exclude "
         "landing pad regions. If forced programmatically, the pad's cells are not overwritten."),
        ("<b>Drone flies through an unscanned obstacle:</b>",
         "Possible if the obstacle is in unexplored terrain and the drone's sensor hasn't revealed it yet. "
         "In practice, the sensor's 2.5 m range usually detects obstacles before the drone reaches them, "
         "especially with the directional 90° FOV pointing forward."),
        ("<b>Closing the viewer window:</b>",
         "The main loop exits gracefully. Trajectory logs are flushed, and the sensor filtering "
         "analysis graph is generated and saved before Python exits."),
    ]
    for title, desc in edge_cases:
        S.append(Paragraph(title, ParagraphStyle("edge_t", parent=BODY, fontName="Helvetica-Bold", spaceAfter=1)))
        S.append(Paragraph(desc, ParagraphStyle("edge_d", parent=BODY, leftIndent=12, spaceAfter=6)))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  14. Troubleshooting & Debugging
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("14. Troubleshooting Guide", H1))

    issues = [
        ("<b>Problem: Viewer opens but keypresses don't work</b>",
         "<b>Solution:</b> Click inside the MuJoCo viewer window first. The window must have OS-level input "
         "focus to receive keyboard events."),
        ("<b>Problem: 'scene.xml not found' error</b>",
         "<b>Solution:</b> Ensure you run the simulation from the project directory. All file paths are relative."),
        ("<b>Problem: 'drone_joint not found' error</b>",
         "<b>Solution:</b> The scene.xml file may be corrupted. Re-run <font face='Courier'>add_rocks.py</font> "
         "to regenerate the dynamic stones block. Ensure the drone body is intact."),
        ("<b>Problem: Drone oscillates or flips during hover</b>",
         "<b>Solution:</b> Check if the drone mass is correct (expected ~1.8 kg). The PID gains are tuned "
         "for this mass. If scene.xml was modified, the gains may need adjustment."),
        ("<b>Problem: Path planning fails — 'no path found'</b>",
         "<b>Solution 1:</b> The destination may be surrounded by obstacles. Try scanning more terrain first (C key). "
         "<b>Solution 2:</b> Dynamic obstacles may have blocked all routes. Wait for some to be recycled."),
        ("<b>Problem: Dynamic obstacles don't appear visually</b>",
         "<b>Solution:</b> The obstacle bodies may not have been loaded from scene.xml. Ensure "
         "<font face='Courier'>dynamic_obstacle_1</font> through <font face='Courier'>dynamic_obstacle_5</font> "
         "exist in the XML file inside <font face='Courier'>&lt;worldbody&gt;</font>."),
        ("<b>Problem: sensor_filtering_analysis.png is not generated</b>",
         "<b>Solution:</b> The graph is generated only when the simulation exits normally (close the viewer or Ctrl+C). "
         "Force-killing the process skips the cleanup. Ensure matplotlib is installed."),
        ("<b>Problem: ImportError for terrain_scanner, dynamic_obstacles, or subsumption</b>",
         "<b>Solution:</b> These files must be in the same directory as simulate.py. Verify all .py files are present."),
        ("<b>Problem: ReportLab not found when generating PDFs</b>",
         "<b>Solution:</b> Install it: <font face='Courier'>pip install reportlab</font>."),
    ]
    for title, desc in issues:
        S.append(Paragraph(title, ParagraphStyle("iss_t", parent=BODY, fontName="Helvetica-Bold", spaceAfter=1)))
        S.append(Paragraph(desc, ParagraphStyle("iss_d", parent=BODY, leftIndent=12, spaceAfter=8)))
    S.append(PageBreak())

    # ════════════════════════════════════════════════════════════
    #  15. Appendix: State Machine Reference
    # ════════════════════════════════════════════════════════════
    S.append(Paragraph("15. Appendix", H1))
    S.append(Paragraph(
        "The drone autopilot operates as a finite state machine. The following table documents every "
        "valid state, its entry conditions, active behaviour, and exit transitions.", BODY))
    sm_data = [
        ["State", "Description", "Active Behaviour", "Transitions To"],
        ["LANDED", "Motors off, on ground", "None (idle)", "TAKEOFF (L key)"],
        ["TAKEOFF", "Climbing to hover height", "Navigation (P2)", "HOVER (reached altitude or path hidden)"],
        ["HOVER", "Holding position at altitude", "Navigation (P2)",
         "SCANNING (C key), GOTO_TARGET (S key + target), LANDING_HOME (L key)"],
        ["SCANNING", "360° yaw spin for mapping", "Terrain Scanning (P1)", "HOVER (scan complete)"],
        ["GOTO_TARGET", "Following planned path", "Navigation (P2) or Avoidance (P3)",
         "AUTOLAND (arrived), HOVER (S key / path lost)"],
        ["AUTOLAND", "Descending onto target pad", "Navigation (P2)", "LANDED (touchdown + 2s wait)"],
        ["LANDING_HOME", "Emergency descent in-place", "Emergency Landing (P4)", "LANDED (ground contact)"],
    ]
    t = make_table(sm_data, colWidths=[70, 100, 100, 250])
    S.append(t)
    S.append(Spacer(1, 10))
    S.append(Paragraph(
        "<b>Note:</b> The Kill Switch (K key) can transition from ANY state to LANDED instantly by "
        "activating the Priority 5 behaviour. This overrides all other transitions.", NOTE))

    # ── Build ──────────────────────────────────────────────────
    doc.build(S)
    print(f"[PDF] User manual generated: {os.path.abspath(pdf_path)}")


if __name__ == "__main__":
    build_manual()
