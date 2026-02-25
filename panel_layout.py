"""
PV Array Layout Documentation & Verification
==============================================
Panel numbering follows the roof annotation convention:
  - "1-N"     = Bank 1, panel N  (SE-facing, 170°)
  - "2-1-N"   = Bank 2, Part 1 (North sub-array), panel N
  - "2-2-N"   = Bank 2, Part 2 (South sub-array), panel N

Roof geometry (from annotated aerial image):
  - The building has two main roof sections separated by a hinge line
  - Left/lower section: Bank 1 (SE-facing, 12 panels)
  - Right/upper section: Bank 2 (West-facing, 14 panels in two groups)

Bank 1 — SE-facing (azimuth 170°, tilt 12°)
  Row config: (5, 4, 3), left-aligned
  Anchor: left upper corner at (532882.50, 6983507.00)

  Local layout (before rotation):
    Row 0 (top):    1-1   1-2   1-3   1-4   1-5
    Row 1 (mid):    1-6   1-7   1-8   1-9
    Row 2 (bot):    1-10  1-11  1-12

  Panel numbering: left-to-right within each row, top row first.

Bank 2, Part 1 — North sub-array (azimuth 260°, tilt 20°)
  Row config: (6, 2), left-aligned
  Anchor: left upper corner at (532885.50, 6983519.50)

  Local layout (before rotation):
    Row 0 (top):    2-1-1  2-1-2  2-1-3  2-1-4  2-1-5  2-1-6
    Row 1 (bot):    2-1-7  2-1-8

  After 260° rotation, rows become vertical columns in the aerial view:
    Column 1 (right in image): 2-1-1 through 2-1-6 (running N→S)
    Column 2 (left in image):  2-1-7, 2-1-8

Bank 2, Part 2 — South sub-array (azimuth 260°, tilt 20°)
  Row config: (4, 2), right-aligned
  Anchor: right upper corner at (532890.50, 6983505.50)

  Local layout (before rotation):
    Row 0 (top):    2-2-1  2-2-2  2-2-3  2-2-4
    Row 1 (bot):    2-2-5  2-2-6

  After 260° rotation, rows become vertical columns in the aerial view:
    Column 1 (right in image): 2-2-1 through 2-2-4 (running N→S)
    Column 2 (left in image):  2-2-5, 2-2-6

Coordinate system notes:
  - All coordinates are ETRS-TM35FIN (EPSG:3067)
  - generate_pv_array_points() builds panels in a local frame:
      +x = across the panel row (left to right)
      +y = up the roof slope (bottom row to top row)
      +z = normal to panel surface
  - The frame is then rotated by (180 - azimuth)° around Z,
    tilted by tilt° around X, and translated to the anchor corner.
  - For az=170° (Bank 1): +x ≈ east, +y ≈ south (upslope)
  - For az=260° (Bank 2): +x ≈ north, +y ≈ west (upslope)
    This is why "rows" in the local frame appear as vertical columns
    in the aerial view for Bank 2.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PANEL GENERATION (copied from shadow_matrix_simulation for standalone use)
# ============================================================================

def generate_pv_array_points(
    corner_coords, tilt_deg=12, az_deg=170,
    panel_width_m=1.0, panel_height_m=1.6,
    row_configuration=(5, 4, 3), align="left",
):
    tilt_rad = np.radians(tilt_deg)
    rot_z_rad = np.radians(180 - az_deg)
    num_rows = len(row_configuration)
    total_height = (num_rows - 1) * panel_height_m

    local_points = []
    y_steps = np.linspace(total_height, 0.0, num_rows)
    for i, num_panels in enumerate(row_configuration):
        y = y_steps[i]
        row_width = num_panels * panel_width_m
        if align == "right":
            x_start = -(row_width - panel_width_m / 2)
        else:
            x_start = panel_width_m / 2
        for p in range(num_panels):
            local_points.append([x_start + p * panel_width_m, y, 0.0])
    local_points = np.array(local_points)

    R_tilt = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad),  np.cos(tilt_rad)]
    ])
    R_az = np.array([
        [np.cos(rot_z_rad), -np.sin(rot_z_rad), 0],
        [np.sin(rot_z_rad),  np.cos(rot_z_rad), 0],
        [0, 0, 1]
    ])
    rotated = ((R_az @ R_tilt) @ local_points.T).T
    return rotated + np.asarray(corner_coords)


# ============================================================================
# ARRAY DEFINITIONS — SINGLE SOURCE OF TRUTH
# ============================================================================

# Bank 1: SE-facing (170°, tilt 12°)
BANK1 = {
    "name": "Bank 1 (SE)",
    "azimuth_deg": 170,
    "tilt_deg": 12,
    "nominal_power_w": 3960,   # 12 × 330 W
    "sub_arrays": [
        {
            "name": "Bank 1",
            "corner_xy": (532881.51, 6983506.91),  # derived from 1-10 at (532882.00, 6983507.00)
            "row_configuration": (5, 4, 3),
            "align": "left",
            "panel_labels": [
                # Row 0 (top): 5 panels
                "1-1", "1-2", "1-3", "1-4", "1-5",
                # Row 1 (mid): 4 panels
                "1-6", "1-7", "1-8", "1-9",
                # Row 2 (bot): 3 panels
                "1-10", "1-11", "1-12",
            ],
        },
    ],
}

# Bank 2: West-facing (260°, tilt 20°)
BANK2 = {
    "name": "Bank 2 (West)",
    "azimuth_deg": 260,
    "tilt_deg": 20,
    "nominal_power_w": 4620,   # 14 × 330 W
    "sub_arrays": [
        {
            "name": "Part 1 — North (6+2)",
            "corner_xy": (532882.93, 6983518.73),  # derived from 2-1-1 at (532884.50, 6983518.50)
            "row_configuration": (6, 2),
            "align": "left",
            "panel_labels": [
                # Row 0 (6 panels): appears as right column in aerial view
                "2-1-1", "2-1-2", "2-1-3", "2-1-4", "2-1-5", "2-1-6",
                # Row 1 (2 panels): appears as left column in aerial view
                "2-1-7", "2-1-8",
            ],
        },
        {
            "name": "Part 2 — South (4+2)",
            "corner_xy": (532888.63, 6983503.79),  # derived from 2-2-1 at (532889.50, 6983507.50)
            "row_configuration": (4, 2),
            "align": "right",
            "panel_labels": [
                # Row 0 (4 panels): appears as right column in aerial view
                "2-2-1", "2-2-2", "2-2-3", "2-2-4",
                # Row 1 (2 panels): appears as left column in aerial view
                "2-2-5", "2-2-6",
            ],
        },
    ],
}

# ============================================================================
# SPECIFICATION TABLE
# ============================================================================

SPEC_TABLE = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    PV SYSTEM SPECIFICATIONS                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Panel type: 330 W, 1.0 m × 1.6 m                                     ║
║  Inverter: Fronius Symo 8.2-3-M                                        ║
║                                                                        ║
║  BANK 1 (SE-facing)                                                    ║
║  ├─ Azimuth: 170°    Tilt: 12°                                         ║
║  ├─ Panels: 12       Layout: (5, 4, 3) left-aligned                    ║
║  ├─ Nominal power: 3960 W                                              ║
║  ├─ Inverter channel: MPP1                                              ║
║  └─ Panel 1-10 at: (532882.00, 6983507.00) → corner (532881.51, 6983506.91) ║
║                                                                        ║
║  BANK 2 (West-facing)                                                  ║
║  ├─ Azimuth: 260°    Tilt: 20°                                         ║
║  ├─ Panels: 14       Nominal power: 4620 W                             ║
║  ├─ Inverter channel: MPP2                                              ║
║  ├─ Part 1 — North (8 panels)                                          ║
║  │   ├─ Layout: (6, 2) left-aligned                                    ║
║  │   └─ Panel 2-1-1 at: (532884.50, 6983518.50) → corner (532882.93, 6983518.73) ║
║  └─ Part 2 — South (6 panels)                                          ║
║      ├─ Layout: (4, 2) right-aligned                                   ║
║      └─ Panel 2-2-1 at: (532889.50, 6983507.50) → corner (532888.63, 6983503.79) ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# VERIFICATION: GENERATE & PLOT ALL PANEL POSITIONS
# ============================================================================

def verify_and_plot(target_z=95.5):
    """Generate all panel world coordinates and plot a top-down map."""
    print(SPEC_TABLE)

    all_points = {}

    for bank_def in [BANK1, BANK2]:
        az = bank_def["azimuth_deg"]
        tilt = bank_def["tilt_deg"]
        print(f"\n{bank_def['name']} — az={az}°, tilt={tilt}°")

        for sa in bank_def["sub_arrays"]:
            xy = sa["corner_xy"]
            corner_3d = np.array([xy[0], xy[1], target_z])
            pts = generate_pv_array_points(
                corner_3d,
                tilt_deg=tilt, az_deg=az,
                row_configuration=sa["row_configuration"],
                align=sa["align"],
            )
            labels = sa["panel_labels"]
            assert len(pts) == len(labels), (
                f"{sa['name']}: {len(pts)} points but {len(labels)} labels"
            )

            print(f"\n  {sa['name']} — {len(pts)} panels, "
                  f"config={sa['row_configuration']}, align={sa['align']}")
            print(f"  Anchor: ({xy[0]:.2f}, {xy[1]:.2f})")

            for i, (pt, lbl) in enumerate(zip(pts, labels)):
                all_points[lbl] = pt
                print(f"    {lbl:8s}  X={pt[0]:.2f}  Y={pt[1]:.2f}  Z={pt[2]:.2f}")

    # --- Top-down plot ---
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = {
        "1": "#2ecc71",     # Bank 1: green
        "2-1": "#3498db",   # Bank 2 North: blue
        "2-2": "#e67e22",   # Bank 2 South: orange
    }

    for lbl, pt in all_points.items():
        if lbl.startswith("2-2"):
            c = colors["2-2"]
        elif lbl.startswith("2-1"):
            c = colors["2-1"]
        else:
            c = colors["1"]

        ax.plot(pt[0], pt[1], "s", color=c, markersize=10, markeredgecolor="k",
                markeredgewidth=0.5)
        ax.annotate(lbl, (pt[0], pt[1]), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")

    # Anchor points
    for bank_def in [BANK1, BANK2]:
        for sa in bank_def["sub_arrays"]:
            xy = sa["corner_xy"]
            ax.plot(xy[0], xy[1], "x", color="red", markersize=12, markeredgewidth=2)
            ax.annotate(f"Anchor\n{sa['name']}", (xy[0], xy[1]),
                       fontsize=6, color="red", ha="center", va="top",
                       xytext=(0, -8), textcoords="offset points")

    ax.set_xlabel("Easting (m)", fontsize=11)
    ax.set_ylabel("Northing (m)", fontsize=11)
    ax.set_title("PV Panel Positions — Top-Down View", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2ecc71",
               markersize=10, label="Bank 1 (SE, 170°)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db",
               markersize=10, label="Bank 2-1 North (W, 260°)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e67e22",
               markersize=10, label="Bank 2-2 South (W, 260°)"),
        Line2D([0], [0], marker="x", color="red", markersize=10,
               label="Anchor corners"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig("panel_layout_verification.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(f"\nTotal panels: {len(all_points)}")
    print(f"  Bank 1: {sum(1 for k in all_points if k.startswith('1-'))}")
    print(f"  Bank 2-1: {sum(1 for k in all_points if k.startswith('2-1-'))}")
    print(f"  Bank 2-2: {sum(1 for k in all_points if k.startswith('2-2-'))}")


if __name__ == "__main__":
    verify_and_plot()