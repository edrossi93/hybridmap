"""Contact size in the two acquisitions, and what the mixed-volume share follows.

The depth axis of the depth-resolved traces is the indenter displacement z, the
instrument's depth channel after frame-compliance and drift correction. The
contact depth is smaller:

    h_c = z - 0.75 P / S                    (Oliver and Pharr, Berkovich)

and the equivalent contact radius of a Berkovich contact is

    a = sqrt(24.5 / pi) h_c

This script writes, for §3.3, §4.1, §4.3.6 and §4.4:

  * h_c and a per class, in the load-controlled map at 0.53 mN (from its own
    contact-depth channel) and in the depth-resolved map at z = 80, 150 and
    300 nm (from the trace, class medians);
  * the classification of the basalt-centre band re-read at several
    displacements with the same 15 and 40 GPa cuts, so the mixed-volume,
    matrix and basalt shares can be followed with depth, and the depth at
    which the contact depth of the stiff contacts matches that of the
    load-controlled stiff contacts;
  * the mixed-volume share of the load-controlled basalt-rich zone on the
    same 40 GPa upper cut;
  * the reading of the §4.4 stiff contacts at the displacement where their
    contact depth equals the load-controlled contact depth on basalt;
  * whether the fall of the mixed-volume contacts between 100 and 300 nm
    follows the distance to the nearest compliant contact.

Run: python contact_geometry.py
Writes: data/contact_geometry.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from classify import E_LOWER, E_UPPER_DEPTH, E_UPPER_SHALLOW      # noqa: E402
from figS7_basalt_depth import SELECT_AT_NM, SELECT_ABOVE_GPA      # noqa: E402

RADIUS_FACTOR = float(np.sqrt(24.5 / np.pi))
CLASSES = ["basalt", "mixed-volume", "flax", "matrix"]
BAND = "basalt-centre"
REREAD_AT = (50.0, 55.0, 60.0, 70.0, 80.0, 100.0, 150.0, 200.0, 300.0)
NEAR_UM = 3.0          # one grid pitch of the depth-resolved map


def contact_depth(curves):
    c = curves.copy()
    c["h_c_nm"] = c.depth_nm - 0.75 * (c.load_mN * 1e-3) / c.stiffness_N_per_m * 1e9
    return c


def reread(E):
    """The rule of §2.3 inside the basalt-centre band, at any displacement."""
    return np.where(E < E_LOWER, "matrix",
                    np.where(E < E_UPPER_DEPTH, "mixed-volume", "basalt"))


def main():
    shallow = pd.read_csv(DATA / "classified_shallow.csv")
    depth = pd.read_csv(DATA / "classified_depth_resolved.csv")
    curves = contact_depth(pd.read_csv(DATA / "depth_resolved_curves.csv"))
    kept = depth[depth.constituent != "QC/invalid"]
    lab = kept.set_index("indent_id").constituent

    out = {"radius_factor": RADIUS_FACTOR}

    # ---- load-controlled contacts at 0.53 mN ------------------------------
    lc = {}
    for c in CLASSES:
        s = shallow[shallow.constituent == c]
        h = float(s.contact_depth_nm.median())
        lc[c] = {"n": int(len(s)), "h_c_nm": h, "a_um": RADIUS_FACTOR * h / 1e3,
                 "h_max_nm": float(s.max_depth_nm.median())}
    out["load_controlled"] = lc

    # ---- depth-resolved contacts at fixed displacement --------------------
    dr = {}
    cv = curves[curves.indent_id.isin(lab.index)].copy()
    cv["cls"] = cv.indent_id.map(lab)
    for z in (80.0, 150.0, 300.0):
        at = cv[np.isclose(cv.depth_nm, z)]
        dr[f"{z:g}"] = {c: {"h_c_nm": float(at[at.cls == c].h_c_nm.median()),
                           "z_minus_h_c_nm": float((z - at[at.cls == c].h_c_nm).median()),
                           "a_um": RADIUS_FACTOR * float(at[at.cls == c].h_c_nm.median()) / 1e3}
                       for c in CLASSES}
    out["depth_resolved_by_displacement"] = dr

    # ---- the band re-read with depth --------------------------------------
    band_ids = kept.loc[kept.zone == BAND, "indent_id"]
    bc = cv[cv.indent_id.isin(band_ids)]
    shares = {}
    for z in REREAD_AT:
        at = bc[np.isclose(bc.depth_nm, z)].dropna(subset=["E_GPa"])
        r = reread(at.E_GPa.to_numpy())
        stiff = at[r != "matrix"]
        shares[f"{z:g}"] = {
            "n": int(len(at)),
            "mixed_volume_pct": 100.0 * float(np.mean(r == "mixed-volume")),
            "matrix_pct": 100.0 * float(np.mean(r == "matrix")),
            "basalt_pct": 100.0 * float(np.mean(r == "basalt")),
            "h_c_stiff_nm": float(stiff.h_c_nm.median())}
    out["basalt_centre_reread"] = shares

    # the load-controlled basalt-rich zone on the same 40 GPa upper cut
    z = shallow[(shallow.zone == "basalt") & (shallow.constituent != "QC/invalid")]
    mixed40 = (z.E_GPa >= E_LOWER) & (z.E_GPa < E_UPPER_DEPTH)
    stiff_lc = z[z.E_GPa >= E_LOWER]
    out["load_controlled_basalt_zone"] = {
        "n": int(len(z)),
        "mixed_volume_pct_at_40": 100.0 * float(mixed40.mean()),
        "mixed_volume_pct_at_70": 100.0 * float(
            ((z.E_GPa >= E_LOWER) & (z.E_GPa < E_UPPER_SHALLOW)).mean()),
        "h_c_stiff_nm": float(stiff_lc.contact_depth_nm.median())}

    # ---- the §4.4 stiff contacts at matched contact depth ------------------
    sel80 = bc[np.isclose(bc.depth_nm, SELECT_AT_NM)]
    sel = sel80.loc[sel80.E_GPa > SELECT_ABOVE_GPA, "indent_id"]
    s = bc[bc.indent_id.isin(sel)]
    target = lc["basalt"]["h_c_nm"]
    per_z = s.groupby("depth_nm").agg(h_c=("h_c_nm", "median"),
                                      E=("E_GPa", "median"),
                                      P=("load_mN", "median"),
                                      n=("E_GPa", "count"))
    zm = float((per_z.h_c - target).abs().idxmin())
    out["stiff_contacts_matched"] = {
        "n_selected": int(len(sel)),
        "load_controlled_basalt_h_c_nm": target,
        "load_controlled_basalt_load_mN": float(
            shallow.loc[shallow.constituent == "basalt", "peak_load_mN"].median()),
        "z_nm": zm, "h_c_nm": float(per_z.loc[zm, "h_c"]),
        "E_GPa": float(per_z.loc[zm, "E"]), "load_mN": float(per_z.loc[zm, "P"]),
        "n_with_reading": int(per_z.loc[zm, "n"]),
        "a_um_at_300": RADIUS_FACTOR * float(per_z.loc[300.0, "h_c"]) / 1e3}

    # ---- mixed-volume fall against distance to compliant contacts ---------
    b = kept[kept.zone == BAND].set_index("indent_id")
    E = cv.pivot(index="indent_id", columns="depth_nm", values="E_GPa")
    comp = b[E.reindex(b.index)[SELECT_AT_NM] < E_LOWER]
    tree = cKDTree(comp[["x_um", "y_um"]].to_numpy())
    mv = b[b.constituent == "mixed-volume"]
    ratio = (E.reindex(mv.index)[300.0] / E.reindex(mv.index)[100.0])
    dist, _ = tree.query(mv[["x_um", "y_um"]].to_numpy(), k=2)
    # a mixed-volume contact reading below 15 GPa at 80 nm is its own nearest
    # compliant contact; take the next one in that case
    self_hit = dist[:, 0] < 1e-9
    d = np.where(self_hit, dist[:, 1], dist[:, 0])
    ok = ratio.notna().to_numpy()
    rho, p = stats.spearmanr(d[ok], ratio.to_numpy()[ok])
    near = ok & (d <= NEAR_UM + 1e-3)
    out["mixed_volume_fall_vs_distance"] = {
        "n": int(ok.sum()), "near_um": NEAR_UM,
        "median_ratio_300_100_near": float(np.median(ratio.to_numpy()[near])),
        "n_near": int(near.sum()),
        "median_ratio_300_100_far": float(np.median(ratio.to_numpy()[ok & ~near])),
        "n_far": int((ok & ~near).sum()),
        "spearman_rho": float(rho), "p": float(p)}

    path = DATA / "contact_geometry.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf8",
                    newline="\n")

    print("contact depth and equivalent radius, class medians")
    print("  load-controlled, 0.53 mN")
    for c in CLASSES:
        v = lc[c]
        print(f"    {c:<13} h_c {v['h_c_nm']:6.1f} nm  a {v['a_um']:.2f} um  "
              f"h_max {v['h_max_nm']:6.1f} nm")
    for zk, row in dr.items():
        print(f"  depth-resolved, z = {zk} nm")
        for c in CLASSES:
            v = row[c]
            print(f"    {c:<13} h_c {v['h_c_nm']:6.1f} nm  z - h_c "
                  f"{v['z_minus_h_c_nm']:5.1f} nm  a {v['a_um']:.2f} um")
    print(f"\nbasalt-centre band re-read with the {E_LOWER:g}/{E_UPPER_DEPTH:g} GPa cuts")
    for zk, v in shares.items():
        print(f"  z = {zk:>3} nm  n {v['n']:3d}  mixed {v['mixed_volume_pct']:4.1f} %  "
              f"matrix {v['matrix_pct']:4.1f} %  basalt {v['basalt_pct']:4.1f} %  "
              f"h_c stiff {v['h_c_stiff_nm']:5.1f} nm")
    v = out["load_controlled_basalt_zone"]
    print(f"  load-controlled basalt-rich zone: mixed {v['mixed_volume_pct_at_40']:.1f} % "
          f"at 40 GPa, {v['mixed_volume_pct_at_70']:.1f} % at 70 GPa, "
          f"h_c stiff {v['h_c_stiff_nm']:.1f} nm")
    v = out["stiff_contacts_matched"]
    print(f"\nthe {v['n_selected']} stiff contacts at h_c = {v['h_c_nm']:.0f} nm "
          f"(z = {v['z_nm']:g} nm, load {v['load_mN']:.2f} mN): median "
          f"{v['E_GPa']:.1f} GPa on {v['n_with_reading']}; load-controlled basalt "
          f"h_c {v['load_controlled_basalt_h_c_nm']:.0f} nm at "
          f"{v['load_controlled_basalt_load_mN']:.2f} mN; radius at 300 nm "
          f"{v['a_um_at_300']:.2f} um")
    v = out["mixed_volume_fall_vs_distance"]
    print(f"mixed-volume E(300)/E(100): {v['median_ratio_300_100_near']:.2f} within "
          f"{v['near_um']:g} um of a compliant contact (n {v['n_near']}), "
          f"{v['median_ratio_300_100_far']:.2f} beyond (n {v['n_far']}); "
          f"Spearman rho {v['spearman_rho']:.2f}, p {v['p']:.2g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
