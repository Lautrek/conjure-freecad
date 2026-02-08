"""FreeCAD adapter module."""

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from conjure_engine import ConjureConnectionError, ConjureEngine

# Command type constants (use server-provided IDs in production)
_CT = {
    "c1": "create_box",
    "c2": "create_cylinder",
    "c3": "create_sphere",
    "c4": "create_cone",
    "c5": "create_torus",
    "c6": "create_polygon_prism",
    "b1": "boolean_fuse",
    "b2": "boolean_cut",
    "b3": "boolean_intersect",
    "t1": "move_object",
    "t2": "rotate_object",
    "t3": "copy_object",
    "t4": "delete_object",
    "m1": "create_fillet",
    "m2": "create_chamfer",
    "q1": "get_state",
    "q2": "get_object_details",
    "q3": "find_objects",
    "q4": "list_faces",
    "q5": "list_edges",
    "q6": "get_bounding_box",
    "q7": "measure_distance",
    "e1": "export_stl",
    "e2": "export_step",
    "e3": "import_stl",
    "v1": "capture_view",
    "v2": "set_view",
    "x1": "run_script",
    "x2": "eval_expression",
}
_CT_REV = {v: k for k, v in _CT.items()}


@dataclass
class AdapterResult:
    """Operation result."""

    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None


class FreeCADAdapter:
    """CAD adapter."""

    def __init__(self, host: str = "localhost", port: int = 9876):
        self.engine = ConjureEngine(host, port)

    async def execute(self, command_type: str, params: Dict[str, Any]) -> AdapterResult:
        """Execute command."""
        try:
            h = self._h(command_type)
            if h is None:
                return AdapterResult(success=False, data={}, error="Invalid op")
            r = await h(params)
            return AdapterResult(success=True, data=r)
        except ConjureConnectionError:
            return AdapterResult(success=False, data={}, error="Connection error")
        except Exception:
            return AdapterResult(success=False, data={}, error="Execution error")

    def _h(self, ct: str):
        """Handler lookup."""
        return {
            "create_box": self._c1,
            "create_cylinder": self._c2,
            "create_sphere": self._c3,
            "create_polygon_prism": self._c6,
            "boolean_fuse": self._b1,
            "boolean_cut": self._b2,
            "boolean_intersect": self._b3,
            "move_object": self._t1,
            "rotate_object": self._t2,
            "copy_object": self._t3,
            "delete_object": self._t4,
            "get_state": self._q1,
            "get_bounding_box": self._q6,
            "find_objects": self._q3,
            "export_stl": self._e1,
            "export_step": self._e2,
            "set_view": self._v2,
            "capture_view": self._v1,
            "run_script": self._x1,
            "eval_expression": self._x2,
        }.get(ct)

    # Handlers

    async def _c1(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n, ln, w, h = p["name"], p["length"], p["width"], p["height"]
        pos = p.get("position", [0, 0, 0])
        return self.engine.run_script(f"""import FreeCAD; import Part
doc = FreeCAD.ActiveDocument or FreeCAD.newDocument("U")
o = doc.addObject("Part::Box", "{n}"); o.Length = {ln}; o.Width = {w}; o.Height = {h}
o.Placement.Base = FreeCAD.Vector({pos[0]}, {pos[1]}, {pos[2]}); doc.recompute()""")

    async def _c2(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n, r, h = p["name"], p["radius"], p["height"]
        pos = p.get("position", [0, 0, 0])
        return self.engine.run_script(f"""import FreeCAD; import Part
doc = FreeCAD.ActiveDocument or FreeCAD.newDocument("U")
o = doc.addObject("Part::Cylinder", "{n}"); o.Radius = {r}; o.Height = {h}
o.Placement.Base = FreeCAD.Vector({pos[0]}, {pos[1]}, {pos[2]}); doc.recompute()""")

    async def _c3(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n, r = p["name"], p["radius"]
        pos = p.get("position", [0, 0, 0])
        return self.engine.run_script(f"""import FreeCAD; import Part
doc = FreeCAD.ActiveDocument or FreeCAD.newDocument("U")
o = doc.addObject("Part::Sphere", "{n}"); o.Radius = {r}
o.Placement.Base = FreeCAD.Vector({pos[0]}, {pos[1]}, {pos[2]}); doc.recompute()""")

    async def _c6(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n, s, r, h = p["name"], p["sides"], p["radius"], p["height"]
        c = p.get("circumscribed", False)
        pos = p.get("position", [0, 0, 0])
        ar = r if c else r / math.cos(math.pi / s)
        return self.engine.run_script(f"""import FreeCAD; import Part; import math
doc = FreeCAD.ActiveDocument or FreeCAD.newDocument("U")
pts = [FreeCAD.Vector({ar}*math.cos(i*2*math.pi/{s}), {ar}*math.sin(i*2*math.pi/{s}), 0) for i in range({s})]
pts.append(pts[0])
w = Part.makePolygon(pts); f = Part.Face(w); pr = f.extrude(FreeCAD.Vector(0,0,{h}))
o = doc.addObject("Part::Feature", "{n}"); o.Shape = pr
o.Placement.Base = FreeCAD.Vector({pos[0]},{pos[1]},{pos[2]}); doc.recompute()""")

    async def _b1(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n, objs = p["name"], p["objects"]
        os = ",".join([f'doc.getObject("{o}")' for o in objs])
        return self.engine.run_script(f"""import FreeCAD
doc = FreeCAD.ActiveDocument; ss = [o.Shape for o in [{os}] if o]
r = ss[0]
for s in ss[1:]: r = r.fuse(s)
o = doc.addObject("Part::Feature", "{n}"); o.Shape = r; doc.recompute()""")

    async def _b2(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n, b, t = p["name"], p["base"], p["tool"]
        kt = p.get("keep_tool", False)
        return self.engine.run_script(f"""import FreeCAD
doc = FreeCAD.ActiveDocument; bo = doc.getObject("{b}"); to = doc.getObject("{t}")
if bo and to:
    o = doc.addObject("Part::Feature", "{n}"); o.Shape = bo.Shape.cut(to.Shape)
    if not {kt}: doc.removeObject("{t}")
    doc.recompute()""")

    async def _b3(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n, objs = p["name"], p["objects"]
        os = ",".join([f'doc.getObject("{o}")' for o in objs])
        return self.engine.run_script(f"""import FreeCAD
doc = FreeCAD.ActiveDocument; ss = [o.Shape for o in [{os}] if o]
r = ss[0]
for s in ss[1:]: r = r.common(s)
o = doc.addObject("Part::Feature", "{n}"); o.Shape = r; doc.recompute()""")

    async def _t1(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n = p["name"]
        x, y, z = p.get("x", 0), p.get("y", 0), p.get("z", 0)
        rel = p.get("relative", True)
        if rel:
            return self.engine.run_script(f"""import FreeCAD
doc = FreeCAD.ActiveDocument; o = doc.getObject("{n}")
if o: c = o.Placement.Base; o.Placement.Base = FreeCAD.Vector(c.x+{x},c.y+{y},c.z+{z}); doc.recompute()""")
        return self.engine.run_script(f"""import FreeCAD
doc = FreeCAD.ActiveDocument; o = doc.getObject("{n}")
if o: o.Placement.Base = FreeCAD.Vector({x},{y},{z}); doc.recompute()""")

    async def _t2(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n, ax, ang = p["name"], p["axis"].lower(), p["angle"]
        ctr = p.get("center", [0, 0, 0])
        av = {"x": "(1,0,0)", "y": "(0,1,0)", "z": "(0,0,1)"}.get(ax, "(0,0,1)")
        return self.engine.run_script(f"""import FreeCAD
doc = FreeCAD.ActiveDocument; o = doc.getObject("{n}")
if o:
    rot = FreeCAD.Rotation(FreeCAD.Vector{av}, {ang})
    o.Placement = FreeCAD.Placement(o.Placement.Base, rot*o.Placement.Rotation, FreeCAD.Vector({ctr[0]},{ctr[1]},{ctr[2]}))
    doc.recompute()""")

    async def _t3(self, p: Dict[str, Any]) -> Dict[str, Any]:
        src, nn = p["source"], p["new_name"]
        off = p.get("offset", [0, 0, 0])
        return self.engine.run_script(f"""import FreeCAD; import Part
doc = FreeCAD.ActiveDocument; s = doc.getObject("{src}")
if s:
    c = doc.addObject("Part::Feature", "{nn}"); c.Shape = s.Shape.copy()
    c.Placement.Base = FreeCAD.Vector(s.Placement.Base.x+{off[0]},s.Placement.Base.y+{off[1]},s.Placement.Base.z+{off[2]})
    doc.recompute()""")

    async def _t4(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n = p["name"]
        return self.engine.run_script(f"""import FreeCAD
doc = FreeCAD.ActiveDocument
if doc.getObject("{n}"): doc.removeObject("{n}"); doc.recompute()""")

    async def _q1(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.get_state(verbose=p.get("verbose", False))

    async def _q6(self, p: Dict[str, Any]) -> Dict[str, Any]:
        n = p["name"]
        return self.engine.run_script(f"""import FreeCAD; import json
doc = FreeCAD.ActiveDocument; o = doc.getObject("{n}"); r = {{"e": "Not found"}}
if o and hasattr(o, 'Shape'):
    bb = o.Shape.BoundBox
    r = {{"min": [bb.XMin,bb.YMin,bb.ZMin], "max": [bb.XMax,bb.YMax,bb.ZMax], "sz": [bb.XLength,bb.YLength,bb.ZLength]}}
json.dumps(r)""")

    async def _q3(self, p: Dict[str, Any]) -> Dict[str, Any]:
        pat = p.get("pattern", "*")
        ot = p.get("object_type")
        tf = f'and o.TypeId == "{ot}"' if ot else ""
        return self.engine.run_script(f"""import FreeCAD; import fnmatch; import json
doc = FreeCAD.ActiveDocument; r = []
if doc:
    for o in doc.Objects:
        if fnmatch.fnmatch(o.Name, "{pat}") {tf}: r.append({{"n": o.Name, "t": o.TypeId}})
json.dumps({{"o": r}})""")

    async def _e1(self, p: Dict[str, Any]) -> Dict[str, Any]:
        objs, fp = p["objects"], p["filepath"]
        pr = p.get("precision", 0.1)
        os = ",".join([f'doc.getObject("{o}")' for o in objs])
        return self.engine.run_script(f"""import FreeCAD; import Mesh
doc = FreeCAD.ActiveDocument; ss = [o.Shape for o in [{os}] if o]
if ss:
    r = ss[0]
    for s in ss[1:]: r = r.fuse(s)
    m = Mesh.Mesh(); m.addFacets(r.tessellate({pr})); m.write("{fp}")""")

    async def _e2(self, p: Dict[str, Any]) -> Dict[str, Any]:
        objs, fp = p["objects"], p["filepath"]
        os = ",".join([f'doc.getObject("{o}")' for o in objs])
        return self.engine.run_script(f"""import FreeCAD; import Part
doc = FreeCAD.ActiveDocument; ss = [o.Shape for o in [{os}] if o]
if ss:
    r = ss[0]
    for s in ss[1:]: r = r.fuse(s)
    r.exportStep("{fp}")""")

    async def _v2(self, p: Dict[str, Any]) -> Dict[str, Any]:
        d = p.get("direction", "isometric")
        vm = {
            "top": "ViewTop",
            "bottom": "ViewBottom",
            "front": "ViewFront",
            "back": "ViewRear",
            "left": "ViewLeft",
            "right": "ViewRight",
            "isometric": "ViewIsometric",
        }
        vc = vm.get(d.lower(), "ViewIsometric")
        return self.engine.run_script(f"""import FreeCADGui
v = FreeCADGui.ActiveDocument.ActiveView; v.{vc}(); v.fitAll()""")

    async def _v1(self, p: Dict[str, Any]) -> Dict[str, Any]:
        fp, w, h = p["filepath"], p.get("width", 800), p.get("height", 600)
        return self.engine.run_script(f"""import FreeCADGui
FreeCADGui.ActiveDocument.ActiveView.saveImage("{fp}", {w}, {h})""")

    async def _x1(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.run_script(p["script"])

    async def _x2(self, p: Dict[str, Any]) -> Dict[str, Any]:
        return self.engine.eval_expression(p["expression"])

    def health_check(self) -> bool:
        return self.engine.health_check()
