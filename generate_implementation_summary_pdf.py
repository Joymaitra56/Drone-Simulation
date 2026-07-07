"""
generate_implementation_summary_pdf.py
Generates a professional PDF report: implementation_summary.pdf
Uses reportlab only (no external dependencies beyond what's already installed).
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import datetime

# ─────────────────────────────────────────────
#  Color Palette
# ─────────────────────────────────────────────
C_BG_DARK    = HexColor("#0D1117")
C_ACCENT     = HexColor("#00FF88")
C_ACCENT2    = HexColor("#00BFFF")
C_HEADER_BG  = HexColor("#161B22")
C_ROW_ALT    = HexColor("#1A2332")
C_ROW_EVEN   = HexColor("#0F1A26")
C_TEXT_MAIN  = HexColor("#E6EDF3")
C_TEXT_DIM   = HexColor("#8B949E")
C_BORDER     = HexColor("#30363D")
C_PRIORITY_1 = HexColor("#FF4444")
C_PRIORITY_2 = HexColor("#FF8800")
C_PRIORITY_3 = HexColor("#FFCC00")
C_PRIORITY_4 = HexColor("#00CCFF")
C_PRIORITY_5 = HexColor("#00FF88")
C_WHITE      = HexColor("#FFFFFF")

PAGE_W, PAGE_H = A4

OUTPUT_FILE = "implementation_summary.pdf"

# ─────────────────────────────────────────────
#  Custom Page Template (dark background + header/footer)
# ─────────────────────────────────────────────
class DarkPageCanvas:
    """Adds dark background, header bar, footer, and page number to every page."""

    def __init__(self, filename, doc):
        self.filename = filename
        self.doc      = doc

    def __call__(self, canv, doc):
        canv.saveState()

        # Dark background
        canv.setFillColor(C_BG_DARK)
        canv.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

        # Top header bar
        canv.setFillColor(C_HEADER_BG)
        canv.rect(0, PAGE_H - 38, PAGE_W, 38, fill=1, stroke=0)

        # Accent left stripe
        canv.setFillColor(C_ACCENT)
        canv.rect(0, PAGE_H - 38, 4, 38, fill=1, stroke=0)

        # Header text — project name
        canv.setFillColor(C_ACCENT)
        canv.setFont("Helvetica-Bold", 9)
        canv.drawString(14, PAGE_H - 24, "MuJoCo Drone Simulation")
        canv.setFillColor(C_TEXT_DIM)
        canv.setFont("Helvetica", 8)
        canv.drawString(14, PAGE_H - 34, "Implementation Summary  |  Autonomous Drone Research Project")

        # Header right — date
        today = datetime.date.today().strftime("%B %d, %Y")
        canv.setFillColor(C_TEXT_DIM)
        canv.setFont("Helvetica", 8)
        canv.drawRightString(PAGE_W - 14, PAGE_H - 24, today)

        # Bottom footer bar
        canv.setFillColor(C_HEADER_BG)
        canv.rect(0, 0, PAGE_W, 24, fill=1, stroke=0)

        # Accent bottom stripe
        canv.setFillColor(C_ACCENT)
        canv.rect(0, 0, PAGE_W, 2, fill=1, stroke=0)

        # Footer text
        canv.setFillColor(C_TEXT_DIM)
        canv.setFont("Helvetica", 7.5)
        canv.drawString(14, 8, "Confidential – Research Use Only")
        canv.drawRightString(PAGE_W - 14, 8, f"Page {doc.page}")

        canv.restoreState()


# ─────────────────────────────────────────────
#  Style Definitions
# ─────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=32,
        textColor=C_ACCENT,
        spaceAfter=6,
        alignment=TA_CENTER,
        leading=38,
    )
    styles["cover_subtitle"] = ParagraphStyle(
        "cover_subtitle",
        fontName="Helvetica",
        fontSize=14,
        textColor=C_TEXT_DIM,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    styles["cover_badge"] = ParagraphStyle(
        "cover_badge",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=C_ACCENT2,
        spaceAfter=2,
        alignment=TA_CENTER,
    )
    styles["section_heading"] = ParagraphStyle(
        "section_heading",
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=C_ACCENT,
        spaceBefore=18,
        spaceAfter=6,
        borderPad=4,
    )
    styles["sub_heading"] = ParagraphStyle(
        "sub_heading",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=C_ACCENT2,
        spaceBefore=10,
        spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9,
        textColor=C_TEXT_MAIN,
        spaceAfter=5,
        leading=14,
        alignment=TA_JUSTIFY,
    )
    styles["body_dim"] = ParagraphStyle(
        "body_dim",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=C_TEXT_DIM,
        spaceAfter=4,
        leading=13,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=9,
        textColor=C_TEXT_MAIN,
        spaceAfter=3,
        leading=13,
        leftIndent=14,
        bulletIndent=4,
    )
    styles["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=8,
        textColor=C_ACCENT,
        spaceAfter=3,
        leading=12,
        backColor=C_HEADER_BG,
        leftIndent=8,
        rightIndent=8,
        borderPad=4,
    )
    styles["toc_entry"] = ParagraphStyle(
        "toc_entry",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=C_TEXT_MAIN,
        spaceAfter=4,
        leading=14,
    )
    styles["label_accent"] = ParagraphStyle(
        "label_accent",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=C_ACCENT,
    )
    styles["label_dim"] = ParagraphStyle(
        "label_dim",
        fontName="Helvetica",
        fontSize=8,
        textColor=C_TEXT_DIM,
    )

    return styles


# ─────────────────────────────────────────────
#  Helper: accent-bordered section divider
# ─────────────────────────────────────────────
def section_divider(title, styles):
    return [
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1, color=C_BORDER, spaceAfter=0),
        Paragraph(f'<font color="#00FF88">■</font>  {title}', styles["section_heading"]),
        HRFlowable(width="40%", thickness=1.5, color=C_ACCENT, spaceAfter=6),
    ]


def sub_section(title, styles):
    return [
        Spacer(1, 4),
        Paragraph(f'<font color="#00BFFF">▶</font>  {title}', styles["sub_heading"]),
    ]


# ─────────────────────────────────────────────
#  Table helpers
# ─────────────────────────────────────────────
def dark_table(data, col_widths, header_bg=C_HEADER_BG):
    """Builds a dark-themed table with alternating row colors."""
    style = TableStyle([
        # Header row
        ("BACKGROUND",  (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  C_ACCENT),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  8.5),
        ("BOTTOMPADDING",(0,0), (-1, 0),  6),
        ("TOPPADDING",  (0, 0), (-1, 0),  6),
        # Data rows
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",   (0, 1), (-1, -1), C_TEXT_MAIN),
        ("TOPPADDING",  (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING",(0,1), (-1,-1),  5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",(0, 0), (-1, -1), 8),
        # Grid
        ("GRID",        (0, 0), (-1, -1), 0.4, C_BORDER),
        ("LINEBELOW",   (0, 0), (-1, 0),  1.5, C_ACCENT),
    ])

    # Alternating row colors
    for i in range(1, len(data)):
        bg = C_ROW_ALT if i % 2 == 0 else C_ROW_EVEN
        style.add("BACKGROUND", (0, i), (-1, i), bg)

    return Table(data, colWidths=col_widths, style=style, repeatRows=1)


def priority_color(p):
    colors_map = {5: C_PRIORITY_1, 4: C_PRIORITY_2, 3: C_PRIORITY_3,
                  2.5: C_PRIORITY_4, 2: C_PRIORITY_5, 1: HexColor("#AAAAFF")}
    return colors_map.get(p, C_TEXT_MAIN)


# ─────────────────────────────────────────────
#  Document Content Builder
# ─────────────────────────────────────────────
def build_content(styles):
    story = []
    W = PAGE_W - 4.2*cm   # usable text width

    # ══════════════════════════════════════════
    #  COVER PAGE
    # ══════════════════════════════════════════
    story.append(Spacer(1, 3.5*cm))

    # Logo / Icon block
    badge_data = [[
        Paragraph('<font color="#00FF88" size="18">🚁</font>', styles["cover_title"])
    ]]
    badge_t = Table(badge_data, colWidths=[W])
    story.append(badge_t)
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("MuJoCo Drone Simulation", styles["cover_title"]))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Implementation Summary Report", styles["cover_subtitle"]))
    story.append(Spacer(1, 0.5*cm))

    # Accent line
    story.append(HRFlowable(width="60%", thickness=2, color=C_ACCENT,
                             hAlign="CENTER", spaceAfter=16))

    # Badges
    badge_items = [
        "17 Python Modules",
        "Physics-Based PID Control",
        "Subsumption Architecture",
        "A* / Dijkstra Pathfinding",
        "Frontier Exploration",
        "IMU Trajectory Replay",
        "Multi-Sensor Fusion",
        "Post-Mission Analytics",
    ]
    badge_cells = [[Paragraph(f'<font color="#0D1117">  {b}  </font>', ParagraphStyle(
        "b", fontName="Helvetica-Bold", fontSize=8, textColor=C_BG_DARK,
        backColor=C_ACCENT, borderPad=4, alignment=TA_CENTER))
        for b in badge_items[:4]]]
    badge_cells.append([Paragraph(f'<font color="#0D1117">  {b}  </font>', ParagraphStyle(
        "b2", fontName="Helvetica-Bold", fontSize=8, textColor=C_BG_DARK,
        backColor=C_ACCENT2, borderPad=4, alignment=TA_CENTER))
        for b in badge_items[4:]])

    badge_table = Table(badge_cells, colWidths=[W/4]*4,
                        style=TableStyle([
                            ("LEFTPADDING",  (0,0),(-1,-1), 4),
                            ("RIGHTPADDING", (0,0),(-1,-1), 4),
                            ("TOPPADDING",   (0,0),(-1,-1), 5),
                            ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                            ("ROWBACKGROUNDS",(0,0),(-1,-1),[C_HEADER_BG]),
                        ]))
    story.append(badge_table)
    story.append(Spacer(1, 1.2*cm))

    # Meta info box
    today = datetime.date.today().strftime("%B %d, %Y")
    meta = [
        ["Document Type", "Technical Implementation Summary"],
        ["Project",        "Autonomous Drone Simulation — MuJoCo"],
        ["Date Generated", today],
        ["Modules Covered","17 Python source files"],
        ["Output Files",   "13 runtime-generated data files"],
    ]
    meta_data = [[
        Paragraph(f'<font color="#8B949E">{k}</font>', styles["body_dim"]),
        Paragraph(f'<font color="#E6EDF3">{v}</font>', styles["body"])
    ] for k, v in meta]
    meta_t = Table(meta_data, colWidths=[5.5*cm, W-5.5*cm],
                   style=TableStyle([
                       ("BACKGROUND",   (0,0),(-1,-1), C_HEADER_BG),
                       ("GRID",         (0,0),(-1,-1), 0.4, C_BORDER),
                       ("LEFTPADDING",  (0,0),(-1,-1), 10),
                       ("RIGHTPADDING", (0,0),(-1,-1), 10),
                       ("TOPPADDING",   (0,0),(-1,-1), 6),
                       ("BOTTOMPADDING",(0,0),(-1,-1), 6),
                       ("LINEAFTER",    (0,0),(0,-1),  1.5, C_ACCENT),
                   ]))
    story.append(meta_t)
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  TABLE OF CONTENTS
    # ══════════════════════════════════════════
    story += section_divider("Table of Contents", styles)

    toc = [
        ("1.", "Architecture Overview", "3"),
        ("2.", "Core Flight System", "4"),
        ("3.", "Subsumption Architecture — Behavior Stack", "5"),
        ("4.", "Mapping & Exploration Pipeline", "7"),
        ("5.", "Sensor Suite & Fusion", "9"),
        ("6.", "Mission Management & Post-Mission Analysis", "10"),
        ("7.", "Frontier Detection & Ranking", "11"),
        ("8.", "Trajectory Recording & Auto-Return", "13"),
        ("9.", "Utilities, Assets & Report Generation", "14"),
        ("10.", "Runtime Output Files", "15"),
        ("11.", "Keyboard Controls Reference", "16"),
    ]
    for num, title, pg in toc:
        row_data = [[
            Paragraph(f'<font color="#00FF88">{num}</font>', styles["toc_entry"]),
            Paragraph(f'<font color="#E6EDF3">{title}</font>', styles["toc_entry"]),
            Paragraph(f'<font color="#8B949E">{pg}</font>', styles["toc_entry"]),
        ]]
        t = Table(row_data, colWidths=[1*cm, W-2.2*cm, 1.2*cm],
                  style=TableStyle([
                      ("TOPPADDING",   (0,0),(-1,-1), 4),
                      ("BOTTOMPADDING",(0,0),(-1,-1), 4),
                      ("LINEBELOW",    (0,0),(-1,-1), 0.3, C_BORDER),
                  ]))
        story.append(t)

    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 1: ARCHITECTURE OVERVIEW
    # ══════════════════════════════════════════
    story += section_divider("1. Architecture Overview", styles)
    story.append(Paragraph(
        "The project is a fully autonomous, physics-based drone simulation built on MuJoCo. "
        "It is structured in decoupled, modular layers where each module has a single responsibility. "
        "The main orchestrator <font color='#00FF88'>simulate.py</font> wires all components together "
        "and drives the real-time simulation loop.", styles["body"]))
    story.append(Spacer(1, 8))

    arch_data = [
        ["Layer", "Module(s)", "Role"],
        ["Orchestrator",         "simulate.py",                       "Main loop, state machine, A*/Dijkstra"],
        ["Physics",              "DronePhysicsController",             "Force-based PID altitude + attitude control"],
        ["Autopilot",            "DroneAutopilot",                     "High-level command dispatch + sensor wiring"],
        ["Behavior Control",     "subsumption.py (6 behaviors)",       "Priority-based reactive behavior suppression"],
        ["Sensing",              "sensor_simulation.py",               "GPS, IMU, LiDAR, barometer, magnetometer"],
        ["Fusion",               "sensor_fusion.py",                   "Weighted multi-sensor pose estimation"],
        ["Mapping",              "occupancy_grid_mapping.py",          "Probabilistic SLAM-style occupancy grid"],
        ["Terrain Discovery",    "terrain_scanner.py",                 "Fog-of-war reveal with directional FOV"],
        ["Exploration",          "frontier_detection.py + ranking.py", "Boundary detection + ranked target selection"],
        ["Coverage Analytics",   "coverage_metrics.py",               "Exploration % tracking + auto-return trigger"],
        ["Path Return",          "imu_trajectory.py",                  "Record path outbound, replay reversed to home"],
        ["Mission Control",      "mission_manager.py",                 "High-level mission state lifecycle"],
        ["Post Processing",      "post_mission_analysis.py",           "Offline inter-pad route cost matrix"],
        ["Obstacles",            "dynamic_obstacles.py",               "Moving obstacle grid updates + replan trigger"],
    ]
    story.append(dark_table(arch_data,
        [3.2*cm, 5.5*cm, W - 8.7*cm]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 2: CORE FLIGHT SYSTEM
    # ══════════════════════════════════════════
    story += section_divider("2. Core Flight System", styles)

    story += sub_section("2.1  DronePhysicsController  (simulate.py)", styles)
    story.append(Paragraph(
        "Implements purely MuJoCo-based force injection via <font color='#00FF88'>data.xfrc_applied</font>. "
        "No qpos manipulation is used — the drone is controlled exclusively through applied forces and torques. "
        "The controller runs at every physics step (200 Hz default).", styles["body"]))
    story.append(Spacer(1, 6))

    pid_data = [
        ["Control Axis",  "Type",  "Gains",                          "Notes"],
        ["Altitude (Z)",  "PID",   "KP=28.0  KI=4.5  KD=12.0",      "Integral clamped ±8.0 to prevent windup"],
        ["Horizontal XY", "PD",    "KP=18.0  KD=9.0",                "Max lateral force 30 N; mass-scaled"],
        ["Roll / Pitch",  "PD",    "KP_ATT=6.0  KD_ATT=2.5",        "Quaternion → RPY via closed-form formula"],
        ["Yaw",           "P",     "KP_YAW=3.0",                     "Shortest-angle error; wraps ±π"],
        ["Linear Drag",   "Damp",  "K_DRAG_LIN=1.8",                 "Applied to all 3 velocity axes"],
        ["Rotational Drag","Damp", "K_DRAG_ROT=0.6",                 "Reduces oscillation on attitude axes"],
    ]
    story.append(dark_table(pid_data, [3.2*cm, 1.5*cm, 5.2*cm, W-9.9*cm]))
    story.append(Spacer(1, 8))

    story += sub_section("2.2  A* Pathfinder  (simulate.py → run_astar)", styles)
    story.append(Paragraph(
        "8-connected grid search using the <b>octile distance heuristic</b> (admissible for diagonal moves). "
        "Cells adjacent to occupied cells receive a <font color='#00FF88'>+3.0 inflation cost</font> penalty "
        "to keep the drone away from wall edges. The planner can be switched to Dijkstra at runtime "
        "via the <font color='#00BFFF'>A</font> key.", styles["body"]))
    story.append(Spacer(1, 6))

    astar_data = [
        ["Feature",              "Implementation Detail"],
        ["Heuristic",            "Octile: max(dx,dy) + (√2−1)·min(dx,dy)  — admissible, consistent"],
        ["Connectivity",         "8-directional (cardinal 1.0, diagonal 1.414)"],
        ["Obstacle Inflation",   "+3.0 extra cost for cells within 1-cell of any obstacle"],
        ["Data Structure",       "Python heapq min-heap on f_score"],
        ["Fallback",             "Dijkstra (run_dijkstra) — selectable at runtime"],
        ["Grid Resolution",      "160×160 cells over 16×16 m world (0.1 m/cell)"],
    ]
    story.append(dark_table(astar_data, [4.5*cm, W-4.5*cm]))
    story.append(Spacer(1, 8))

    story += sub_section("2.3  DroneAutopilot State Machine  (simulate.py)", styles)
    states_data = [
        ["State",         "Entry Condition",                       "Exit Condition"],
        ["LANDED",        "Power-on / touchdown complete",         "L key pressed → TAKEOFF"],
        ["TAKEOFF",       "L key from LANDED",                    "Altitude within 10 cm of hover_height"],
        ["HOVER",         "Takeoff complete / scan done",          "C key (SCANNING), cmd_goto (GOTO_TARGET)"],
        ["SCANNING",      "C key while HOVER",                    "360° yaw accumulated (2π rad)"],
        ["GOTO_TARGET",   "S key + target pad set",               "Within 30 cm of target pad center"],
        ["AUTOLAND",      "Arrival at target pad",                 "Touchdown + 2 s wait → LANDED"],
        ["LANDING_HOME",  "L key while flying / kill command",    "Terrain height reached → LANDED"],
    ]
    story.append(dark_table(states_data, [2.8*cm, 5.6*cm, W-8.4*cm]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 3: SUBSUMPTION ARCHITECTURE
    # ══════════════════════════════════════════
    story += section_divider("3. Subsumption Architecture — Behavior Stack", styles)
    story.append(Paragraph(
        "Implements Brooks' <b>Subsumption Architecture</b> — a reactive, layered control system "
        "where higher-priority behaviors suppress lower-priority ones. "
        "The <font color='#00FF88'>BehaviorManager.step()</font> method iterates the sorted behavior list "
        "each simulation tick and executes the first behavior whose <font color='#00FF88'>check_trigger()</font> "
        "returns True, skipping all lower-priority behaviors.", styles["body"]))
    story.append(Spacer(1, 8))

    # Priority stack table
    beh_data = [
        ["Priority", "Behavior Class", "Trigger Condition", "Action Taken"],
        ["5  ■", "KillSwitchBehavior",
         "kill_switch_active == True",
         "Zero all forces, state→LANDED, clear path"],
        ["4  ■", "EmergencyLandingBehavior",
         "state == 'LANDING_HOME'",
         "PID descent in-place to terrain height"],
        ["3  ■", "ObstacleAvoidanceBehavior",
         "state=='GOTO_TARGET' AND path_blocked",
         "Re-run A*/Dijkstra, rebuild waypoints"],
        ["2.5 ■","ReturnHomeBehavior",
         "auto_return_active == True",
         "Pull reversed trajectory waypoints, steer home"],
        ["2  ■", "NavigationBehavior",
         "state in (TAKEOFF, HOVER, GOTO_TARGET, AUTOLAND)",
         "Normal takeoff/hover/path-follow/autoland"],
        ["1  ■", "TerrainScanningBehavior",
         "state == 'SCANNING'",
         "Apply spin torque; end at 2π accumulated yaw"],
    ]

    priority_colors = [C_PRIORITY_1, C_PRIORITY_2, C_PRIORITY_3,
                       C_PRIORITY_4, C_PRIORITY_5, HexColor("#AAAAFF")]

    beh_style = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  C_HEADER_BG),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  C_ACCENT),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8.5),
        ("BOTTOMPADDING",(0, 0), (-1, 0),  6),
        ("TOPPADDING",   (0, 0), (-1, 0),  6),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",    (0, 1), (-1, -1), C_TEXT_MAIN),
        ("TOPPADDING",   (0, 1), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("GRID",         (0, 0), (-1, -1), 0.4, C_BORDER),
        ("LINEBELOW",    (0, 0), (-1, 0),  1.5, C_ACCENT),
    ])
    for i, c in enumerate(priority_colors, start=1):
        beh_style.add("BACKGROUND", (0, i), (0, i), c)
        beh_style.add("TEXTCOLOR",  (0, i), (0, i), C_BG_DARK)
        beh_style.add("FONTNAME",   (0, i), (0, i), "Helvetica-Bold")
        bg = C_ROW_ALT if i % 2 == 0 else C_ROW_EVEN
        beh_style.add("BACKGROUND", (1, i), (-1, i), bg)

    beh_table = Table(beh_data,
                      colWidths=[1.5*cm, 4.2*cm, 5.0*cm, W-10.7*cm],
                      style=beh_style, repeatRows=1)
    story.append(beh_table)
    story.append(Spacer(1, 10))

    story += sub_section("3.1  ReturnHomeBehavior — Trajectory Reversal Detail", styles)
    story.append(Paragraph(
        "When <font color='#00FF88'>auto_return_active</font> is set (triggered at 70% coverage or "
        "zero remaining frontiers), this behavior takes over from NavigationBehavior. "
        "It queries <font color='#00FF88'>TrajectoryReplayer.get_next_waypoint(dx, dy)</font> "
        "each tick and issues velocity-mode physics commands toward each reversed waypoint. "
        "On completion (home radius reached), it sets <font color='#00BFFF'>state→LANDING_HOME</font> "
        "which hands off to EmergencyLandingBehavior (priority 4) for final descent.",
        styles["body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 4: MAPPING & EXPLORATION
    # ══════════════════════════════════════════
    story += section_divider("4. Mapping & Exploration Pipeline", styles)

    story += sub_section("4.1  TerrainScanner  (terrain_scanner.py)", styles)
    story.append(Paragraph(
        "Implements a <b>fog-of-war</b> reveal system. The drone sees only what its sensors "
        "have scanned. Two grids are maintained in parallel:", styles["body"]))
    ts_data = [
        ["Grid / Mask",      "Type",        "Purpose"],
        ["discovered_grid",  "int8 array",  "Holds obstacle status for explored cells only"],
        ["explored_mask",    "bool array",  "True if the cell has been seen by the sensor"],
    ]
    story.append(dark_table(ts_data, [3.8*cm, 2.8*cm, W-6.6*cm]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Each step: drone grid position is computed, cells within <font color='#00FF88'>sensor_radius=2.5</font> "
        "grid units are checked against a <font color='#00FF88'>fov_deg=90°</font> forward cone. "
        "Only cells inside the cone are marked as explored. "
        "The planning grid returned to A* always uses only explored cells.",
        styles["body"]))

    story += sub_section("4.2  OccupancyGridMapping  (occupancy_grid_mapping.py)", styles)
    story.append(Paragraph(
        "Probabilistic occupancy mapping using a <b>log-odds update rule</b>. "
        "Each sensor reading increments or decrements the log-odds for a cell. "
        "The grid stores three cell states:", styles["body"]))
    grid_states = [
        ["Value",  "Constant",      "Meaning"],
        ["-1",     "CELL_UNKNOWN",  "Never observed by any sensor"],
        ["0",      "CELL_FREE",     "Observed and confirmed clear"],
        ["1",      "CELL_OCCUPIED", "Obstacle detected"],
    ]
    story.append(dark_table(grid_states, [1.5*cm, 3.5*cm, W-5.0*cm]))

    story += sub_section("4.3  FrontierDetection  (frontier_detection.py)", styles)
    story.append(Paragraph(
        "Identifies the boundary between <font color='#00FF88'>CELL_FREE</font> and "
        "<font color='#8B949E'>CELL_UNKNOWN</font> space using a BFS/flood-fill over the occupancy grid. "
        "Contiguous frontier cells are grouped into labeled <b>regions</b>, each with a centroid "
        "(world + grid coordinates) and a size (cell count). "
        "Regions below a minimum size threshold are discarded as noise.",
        styles["body"]))

    story += sub_section("4.4  CoverageMetrics  (coverage_metrics.py)", styles)
    story.append(Paragraph(
        "Computes exploration progress after every mapping update. "
        "Coverage percentage = (free_cells + occupied_cells) / total_cells × 100. "
        "Provides <font color='#00FF88'>get_statistics()</font> returning a dictionary with "
        "area covered, cell counts by type, elapsed mission time, and current coverage %. "
        "When coverage ≥ 70%, calls <font color='#00FF88'>autopilot._trigger_auto_return()</font>.",
        styles["body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 5: SENSOR SUITE & FUSION
    # ══════════════════════════════════════════
    story += section_divider("5. Sensor Suite & Fusion", styles)

    story += sub_section("5.1  SensorSimulationSuite  (sensor_simulation.py)", styles)
    story.append(Paragraph(
        "Emulates all sensors a real autonomous drone would carry. "
        "Each sensor adds configurable Gaussian noise via NumPy to simulate realistic measurement error.",
        styles["body"]))
    sensor_data = [
        ["Sensor",       "Output",                         "Noise Model"],
        ["GPS",          "x, y, z position",               "Gaussian σ=0.05 m horizontal, 0.1 m vertical"],
        ["IMU",          "acceleration (ax,ay,az), gyro",  "White noise + bias drift"],
        ["Barometer",    "altitude (z)",                   "Gaussian σ=0.02 m"],
        ["LiDAR",        "range distance per ray",         "Random ray miss probability + σ=0.01 m"],
        ["Magnetometer", "heading angle",                  "Gaussian σ=0.5°"],
    ]
    story.append(dark_table(sensor_data, [2.5*cm, 4.5*cm, W-7.0*cm]))
    story.append(Spacer(1, 8))

    story += sub_section("5.2  SensorFusion  (sensor_fusion.py)", styles)
    story.append(Paragraph(
        "Combines GPS and IMU into a single <font color='#00FF88'>FusedState</font> dataclass "
        "(x, y, z, vx, vy, vz, valid). GPS provides position ground truth; "
        "IMU velocity integration fills time gaps between GPS fixes. "
        "Weighted average fusion: GPS weight 0.8, IMU weight 0.2 when both valid. "
        "Output is consumed by <b>FrontierRanking</b> and <b>CoverageMetrics</b>.",
        styles["body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 6: MISSION MANAGEMENT
    # ══════════════════════════════════════════
    story += section_divider("6. Mission Management & Post-Mission Analysis", styles)

    story += sub_section("6.1  MissionManager + MissionState  (mission_manager.py / mission_state.py)", styles)
    story.append(Paragraph(
        "<font color='#00FF88'>MissionState</font> is a Python Enum defining five lifecycle stages. "
        "<font color='#00FF88'>MissionManager</font> validates state transitions against an allowed "
        "adjacency table, preventing illegal jumps (e.g., IDLE→LAND). "
        "A <font color='#00BFFF'>force_state()</font> method is provided for emergency overrides.",
        styles["body"]))
    ms_data = [
        ["State",             "Entered When",                              "Exits To"],
        ["IDLE",              "Simulation starts",                         "EXPLORE"],
        ["EXPLORE",           "E key pressed while hovering",              "RETURN_HOME"],
        ["RETURN_HOME",       "Coverage ≥ 70% or no frontiers remain",     "LAND"],
        ["LAND",              "Trajectory reversal complete, home reached", "MISSION_COMPLETE"],
        ["MISSION_COMPLETE",  "Drone has landed at home pad",              "— (terminal)"],
    ]
    story.append(dark_table(ms_data, [3.2*cm, 5.8*cm, W-9.0*cm]))

    story += sub_section("6.2  PostMissionAnalysis  (post_mission_analysis.py)", styles)
    story.append(Paragraph(
        "Offline analysis executed automatically after the drone lands. "
        "Runs both A* and Dijkstra between <b>every pair</b> of the 6 landing pads "
        "on the final discovered occupancy grid. Produces:", styles["body"]))
    for item in [
        "A route cost matrix (JSON) — inter-pad travel costs for both algorithms",
        "mission_route_map.png — All inter-pad paths rendered on the explored map",
        "Console-printed route matrix via print_route_matrix()",
    ]:
        story.append(Paragraph(f"  •  {item}", styles["bullet"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 7: FRONTIER DETECTION & RANKING
    # ══════════════════════════════════════════
    story += section_divider("7. Frontier Detection & Ranking", styles)

    story += sub_section("7.1  FrontierRanking  (frontier_ranking.py)", styles)
    story.append(Paragraph(
        "Scores every detected frontier region using a <b>four-component weighted formula</b>. "
        "The best-scored frontier becomes the drone's next autonomous exploration target.",
        styles["body"]))
    story.append(Spacer(1, 6))

    score_data = [
        ["Component",        "Weight", "Computation Method",
         "Normalization"],
        ["Information Gain", "40%",
         "Count unique CELL_UNKNOWN neighbors in 8-dirs around all region cells",
         "min(1.0, count / 50)"],
        ["Travel Cost",      "30%",
         "Run A* from drone to frontier centroid; apply exponential decay",
         "exp(−0.05 × A* cost)"],
        ["Safety Score",     "20%",
         "Window search (6-cell radius) for nearest obstacle from centroid",
         "min(1.0, dist / 6.0)"],
        ["Reachability",     "10%",
         "Binary: 1.0 if A* finds a path, 0.0 if unreachable (skip entirely)",
         "0.0 or 1.0"],
    ]
    story.append(dark_table(score_data, [2.8*cm, 1.5*cm, 6.2*cm, W-10.5*cm]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Caching optimization:</b> The ranker stores a cache key of "
        "<font color='#00FF88'>(pos_key, grid_sum, regions_len)</font>. "
        "If all three match the previous call, the full re-evaluation is skipped, "
        "avoiding redundant A* calls at every simulation tick.",
        styles["body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<b>Tie-breaking:</b> Final sort uses <font color='#00FF88'>key=lambda x: (score, −region_id)</font> "
        "in descending order, ensuring deterministic selection when scores are equal.",
        styles["body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 8: TRAJECTORY RECORDING & AUTO-RETURN
    # ══════════════════════════════════════════
    story += section_divider("8. Trajectory Recording & Auto-Return", styles)

    story += sub_section("8.1  TrajectoryRecorder  (imu_trajectory.py)", styles)
    story.append(Paragraph(
        "Records the drone's flight path during the exploration phase. "
        "Sampling is rate-limited by two filters applied before storing a waypoint:",
        styles["body"]))
    traj_rec = [
        ["Filter",             "Value",   "Purpose"],
        ["Temporal sampling",  "10 Hz",   "record_hz=10.0 — at most 10 samples per second"],
        ["Distance filter",    "0.05 m",  "min_dist_m=0.05 — ignores stationary hovering"],
    ]
    story.append(dark_table(traj_rec, [3.5*cm, 2.0*cm, W-5.5*cm]))
    story.append(Spacer(1, 5))
    story.append(Paragraph(
        "Provides <font color='#00FF88'>get_log()</font> returning the full waypoint list and "
        "<font color='#00FF88'>get_total_distance_m()</font> for mission statistics.",
        styles["body"]))

    story += sub_section("8.2  TrajectoryReplayer  (imu_trajectory.py)", styles)
    story.append(Paragraph(
        "Receives the recorded log, <b>reverses the waypoint list</b>, and serves waypoints "
        "one at a time. The drone navigates toward each reversed waypoint using the same "
        "velocity-mode physics commands as normal path following. "
        "Home arrival is detected when the drone enters a "
        "<font color='#00FF88'>home_radius_m=0.40 m</font> circle around the home coordinates.",
        styles["body"]))

    story += sub_section("8.3  Auto-Return Trigger Logic", styles)
    story.append(Paragraph(
        "The auto-return sequence is initiated by "
        "<font color='#00FF88'>DroneAutopilot._trigger_auto_return()</font> "
        "when either condition is met:", styles["body"]))
    for item in [
        "Coverage ≥ 70% (configurable via explore_coverage_threshold)",
        "No reachable frontier regions remain after FrontierRanking update",
    ]:
        story.append(Paragraph(f"  •  {item}", styles["bullet"]))
    story.append(Paragraph(
        "The method stops the recorder, transitions MissionState → RETURN_HOME, "
        "starts the replayer, and sets <font color='#00FF88'>auto_return_active = True</font> "
        "which activates ReturnHomeBehavior (priority 2.5) in the subsumption stack.",
        styles["body"]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 9: UTILITIES & ASSETS
    # ══════════════════════════════════════════
    story += section_divider("9. Utilities, Assets & Report Generation", styles)

    util_data = [
        ["File",                         "Purpose",                                    "Method"],
        ["add_rocks.py",                 "Inject rock obstacles into scene.xml",       "Random pos (pad-zone excluded) → <geom> XML entries"],
        ["generate_assets.py",           "Pre-generate terrain heightmap + grid PNG",  "PIL/NumPy from get_terrain_height() math formula"],
        ["generate_pdf.py",              "Generate project_report.pdf",                "ReportLab A4 multi-section technical report"],
        ["generate_manual_pdf.py",       "Generate drone_operations_manual.pdf",       "ReportLab formatted operations manual"],
        ["run.bat",                      "One-click simulation launcher",              "Batch: activate env → generate_assets.py → simulate.py"],
        ["dynamic_obstacles.py",         "Moving obstacle tracking + replan trigger",  "MuJoCo body positions → grid coords → path_blocked flag"],
    ]
    story.append(dark_table(util_data, [4.0*cm, 5.0*cm, W-9.0*cm]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 10: RUNTIME OUTPUT FILES
    # ══════════════════════════════════════════
    story += section_divider("10. Runtime Output Files", styles)
    story.append(Paragraph(
        "The simulation automatically generates the following data files at runtime. "
        "No manual export is required — all files are written to the project directory.",
        styles["body"]))
    story.append(Spacer(1, 6))

    out_data = [
        ["File",                        "Generated By",             "Contents"],
        ["flight_trajectory.csv",       "simulate.py",              "Per-step: time, state, x/y/z, quaternion, velocity, forces"],
        ["detection_log.csv",           "simulate.py",              "Event log: takeoff, hover, landing, scan events"],
        ["ground_map.png",              "simulate.py",              "Fog-of-war explored map rendered as PNG"],
        ["shortest_path_map.png",       "simulate.py",              "Map with A* path overlay drawn on top"],
        ["map_data.json",               "simulate.py",              "Full grid array, pad locations, path coordinates"],
        ["occupancy_grid.csv",          "occupancy_grid_mapping.py","Raw grid cell values (−1 / 0 / 1)"],
        ["map_metadata.csv",            "simulate.py",              "Grid resolution, origin, world dimensions"],
        ["landing_spots.csv",           "simulate.py",              "All pad names, world + grid coordinates"],
        ["visiting_sequence.csv",       "simulate.py",              "Pad visit order during mission"],
        ["sensor_metrics.json",         "sensor_simulation.py",     "Sensor noise statistics and accuracy metrics"],
        ["mission_log.json",            "simulate.py",              "Coverage stats, trajectory snapshot count, end time"],
        ["mission_route_analysis.json", "post_mission_analysis.py", "Inter-pad A*/Dijkstra cost matrix for all pad pairs"],
        ["mission_route_map.png",       "post_mission_analysis.py", "Map with all inter-pad routes visualized"],
    ]
    story.append(dark_table(out_data, [4.8*cm, 4.0*cm, W-8.8*cm]))
    story.append(PageBreak())

    # ══════════════════════════════════════════
    #  SECTION 11: KEYBOARD CONTROLS
    # ══════════════════════════════════════════
    story += section_divider("11. Keyboard Controls Reference", styles)
    story.append(Paragraph(
        "All simulation controls are handled in the MuJoCo viewer key callback. "
        "Controls operate on the drone's state machine and trigger the appropriate "
        "autopilot command methods.", styles["body"]))
    story.append(Spacer(1, 6))

    key_data = [
        ["Key",   "Action",                     "Method Called",              "Precondition"],
        ["L",     "Takeoff",                    "cmd_takeoff()",              "state == LANDED"],
        ["L",     "Return Home + Land",         "cmd_takeoff()",              "state == HOVER/GOTO_TARGET"],
        ["C",     "Start 360° terrain scan",    "cmd_scan()",                 "state == HOVER"],
        ["S",     "Show path + start flight",   "cmd_toggle_path()",          "Path planned via cmd_goto"],
        ["E",     "Enter exploration mode",     "cmd_start_exploration()",    "state == HOVER"],
        ["1–5",   "Fly to Pad 1–5",             "cmd_goto('Pad N')",          "Map available"],
        ["H",     "Fly to Home pad",            "cmd_goto('Home')",           "Map available"],
        ["K",     "Kill switch (motor cut)",    "Sets kill_switch_active",    "Any flight state"],
        ["A",     "Toggle A* ↔ Dijkstra",       "Toggles planner_algorithm",  "Any state"],
    ]
    story.append(dark_table(key_data, [1.2*cm, 4.5*cm, 4.0*cm, W-9.7*cm]))
    story.append(Spacer(1, 12))

    # Final accent divider
    story.append(HRFlowable(width="100%", thickness=1, color=C_BORDER))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f'<font color="#8B949E">Generated automatically on {datetime.date.today().strftime("%B %d, %Y")} '
        f'from project source analysis.  MuJoCo Drone Simulation — Autonomous Drone Research Project.</font>',
        styles["body_dim"]))

    return story


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def generate_pdf():
    print(f"[PDF] Building {OUTPUT_FILE} ...")

    doc = SimpleDocTemplate(
        OUTPUT_FILE,
        pagesize=A4,
        leftMargin=2.2*cm,
        rightMargin=2.0*cm,
        topMargin=2.8*cm,
        bottomMargin=1.8*cm,
        title="MuJoCo Drone Simulation — Implementation Summary",
        author="Autonomous Drone Research Project",
        subject="Implementation Summary",
    )

    styles  = build_styles()
    story   = build_content(styles)
    painter = DarkPageCanvas(OUTPUT_FILE, doc)

    doc.build(story, onFirstPage=painter, onLaterPages=painter)
    print(f"[PDF] DONE - Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_pdf()
