"""sparta/surf.py: nozzle wall + faceplate outline for a DSMC domain starting inside the nozzle."""
import numpy as np

from sparta import surf
from test_breakdown_2d import nozzle_field  # nozzle (r 1 -> 3 mm over x -10..0 mm) + chamber


def test_wall_profile_follows_the_nozzle_wall_from_the_plane_to_the_lip():
    fld = nozzle_field(1e-3, 300.0, 100.0)
    x, r = surf.wall_profile(fld, -6e-3)
    assert x[0] == -6e-3 and x[-1] == 0.0 and np.all(np.diff(x) > 0)
    np.testing.assert_allclose(r, 1e-3 + 2e-3 * (x + 10e-3) / 10e-3, rtol=1e-6)


def test_outline_is_closed_and_its_normals_point_into_the_gas():
    fld = nozzle_field(1e-3, 300.0, 100.0)
    box = (-6e-3, 10e-3, 0.0, 10e-3)
    x_w, r_w = surf.wall_profile(fld, box[0])
    pts = np.array(surf.outline(x_w, r_w, box))
    inside = surf.solid_test(x_w, r_w)
    p1, p2 = pts, np.roll(pts, -1, axis=0)
    mid = 0.5 * (p1 + p2)
    d = p2 - p1
    normal = np.column_stack([-d[:, 1], d[:, 0]]) / np.hypot(*d.T)[:, None]  # SPARTA: z x (p2 - p1)
    in_box = (mid[:, 0] > box[0]) & (mid[:, 0] < box[1]) & (mid[:, 1] > box[2]) & (mid[:, 1] < box[3])
    assert in_box.sum() >= len(x_w) - 1  # the wall lines and the faceplate are in the box
    step = 1e-7
    gas_side = mid[in_box] + step * normal[in_box]
    body_side = mid[in_box] - step * normal[in_box]
    assert not inside(*gas_side.T).any()
    # just behind each line is the body (the faceplate's back is x < 0 above the lip)
    assert inside(*body_side.T).all()


def test_write_surf_round_trip(tmp_path):
    pts = [(0.0, 1.0), (-1.0, 1.0), (-1.0, 2.0), (0.0, 2.0)]
    surf.write_surf(tmp_path / "s.surf", pts, header="test")
    txt = (tmp_path / "s.surf").read_text()
    assert "4 points" in txt and "4 lines" in txt and "\n4 4 1\n" in txt
