"""
Local standards library for FreeCAD workbench.

Loads gear, socket, fastener, material, thread, and profile specifications
from bundled YAML files. Used for offline access to engineering standards.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class GearSpec:
    """Specification for involute gear geometry."""

    id: str
    gear_type: str
    standard: str
    module_mm: float
    num_teeth: int
    pressure_angle_deg: float = 20.0
    face_width_mm: Optional[float] = None
    quality_grade: Optional[int] = None
    helix_angle_deg: Optional[float] = None
    backlash_mm: Optional[float] = None
    profile_shift_coefficient: float = 0.0
    bore_diameter_mm: Optional[float] = None
    description: str = ""
    use_cases: List[str] = field(default_factory=list)
    mating_gears: List[str] = field(default_factory=list)

    # Calculated dimensions
    pitch_diameter_mm: float = field(init=False)
    base_diameter_mm: float = field(init=False)
    addendum_mm: float = field(init=False)
    dedendum_mm: float = field(init=False)
    whole_depth_mm: float = field(init=False)
    working_depth_mm: float = field(init=False)
    clearance_mm: float = field(init=False)
    tooth_thickness_mm: float = field(init=False)
    root_fillet_radius_mm: float = field(init=False)
    tip_diameter_mm: float = field(init=False)
    root_diameter_mm: float = field(init=False)

    def __post_init__(self):
        """Calculate derived dimensions."""
        m = self.module_mm
        z = self.num_teeth
        alpha = math.radians(self.pressure_angle_deg)

        self.pitch_diameter_mm = m * z
        self.base_diameter_mm = self.pitch_diameter_mm * math.cos(alpha)
        self.addendum_mm = m
        self.dedendum_mm = 1.25 * m
        self.whole_depth_mm = 2.25 * m
        self.working_depth_mm = 2.0 * m
        self.clearance_mm = 0.25 * m
        self.tooth_thickness_mm = math.pi * m / 2
        self.root_fillet_radius_mm = 0.38 * m
        self.tip_diameter_mm = self.pitch_diameter_mm + 2 * self.addendum_mm
        self.root_diameter_mm = self.pitch_diameter_mm - 2 * self.dedendum_mm

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "gear_type": self.gear_type,
            "standard": self.standard,
            "module_mm": self.module_mm,
            "num_teeth": self.num_teeth,
            "pressure_angle_deg": self.pressure_angle_deg,
            "pitch_diameter_mm": self.pitch_diameter_mm,
            "base_diameter_mm": self.base_diameter_mm,
            "addendum_mm": self.addendum_mm,
            "dedendum_mm": self.dedendum_mm,
            "whole_depth_mm": self.whole_depth_mm,
            "working_depth_mm": self.working_depth_mm,
            "clearance_mm": self.clearance_mm,
            "tooth_thickness_mm": self.tooth_thickness_mm,
            "root_fillet_radius_mm": self.root_fillet_radius_mm,
            "tip_diameter_mm": self.tip_diameter_mm,
            "root_diameter_mm": self.root_diameter_mm,
            "face_width_mm": self.face_width_mm,
            "quality_grade": self.quality_grade,
            "helix_angle_deg": self.helix_angle_deg,
            "backlash_mm": self.backlash_mm,
            "bore_diameter_mm": self.bore_diameter_mm,
            "description": self.description,
            "use_cases": self.use_cases,
            "mating_gears": self.mating_gears,
        }


class LocalStandardsLibrary:
    """
    Local standards library that loads from bundled YAML files.

    Used when the server is not available or for faster access.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent / "data"

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._formulas: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load a YAML file."""
        if not HAS_YAML:
            return {}
        if not filepath.exists():
            return {}

        with open(filepath) as f:
            return yaml.safe_load(f) or {}

    def load(self, force_reload: bool = False) -> None:
        """Load all YAML data files."""
        if self._loaded and not force_reload:
            return

        self._cache = {
            "sockets": {},
            "fasteners": {},
            "materials": {},
            "threads": {},
            "profiles": {},
            "gears": {},
        }

        # Load each category
        for category in ["sockets", "fasteners", "materials", "threads", "profiles", "gears"]:
            filepath = self.data_dir / f"{category}.yaml"
            if filepath.exists():
                data = self._load_yaml(filepath)

                # Store formulas
                if "formulas" in data:
                    self._formulas[category] = data["formulas"]

                # Store specs
                for spec_id, spec in data.get(category, {}).items():
                    if spec_id.startswith("_"):
                        continue  # Skip guidelines
                    if spec.get("type") == "guidelines":
                        continue

                    spec["id"] = spec_id
                    self._cache[category][spec_id] = spec

        self._loaded = True

    def list_standards(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List available standards by category."""
        self.load()

        if category:
            category = category.lower().rstrip("s")  # Normalize: "gears" -> "gear"
            category_map = {
                "socket": "sockets",
                "fastener": "fasteners",
                "material": "materials",
                "thread": "threads",
                "profile": "profiles",
                "gear": "gears",
            }
            cat_key = category_map.get(category, f"{category}s")

            if cat_key in self._cache:
                return {cat_key: list(self._cache[cat_key].keys())}
            return {}

        return {cat: list(specs.keys()) for cat, specs in self._cache.items()}

    def get_standard(self, category: str, spec_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific standard by category and ID."""
        self.load()

        category = category.lower().rstrip("s")
        category_map = {
            "socket": "sockets",
            "fastener": "fasteners",
            "material": "materials",
            "thread": "threads",
            "profile": "profiles",
            "gear": "gears",
        }
        cat_key = category_map.get(category, f"{category}s")

        spec = self._cache.get(cat_key, {}).get(spec_id)
        if not spec:
            return None

        # For gears, create GearSpec to get calculated dimensions
        if cat_key == "gears":
            try:
                gear = GearSpec(
                    id=spec["id"],
                    gear_type=spec.get("type", "spur"),
                    standard=spec.get("standard", "ISO"),
                    module_mm=spec["module_mm"],
                    num_teeth=spec["num_teeth"],
                    pressure_angle_deg=spec.get("pressure_angle_deg", 20.0),
                    face_width_mm=spec.get("face_width_mm"),
                    quality_grade=spec.get("quality_grade"),
                    helix_angle_deg=spec.get("helix_angle_deg"),
                    backlash_mm=spec.get("backlash_mm"),
                    profile_shift_coefficient=spec.get("profile_shift_coefficient", 0.0),
                    bore_diameter_mm=spec.get("bore_diameter_mm"),
                    description=spec.get("description", ""),
                    use_cases=spec.get("use_cases", []),
                    mating_gears=spec.get("mating_gears", []),
                )
                return gear.to_dict()
            except (KeyError, TypeError):
                pass

        return spec

    def search_standards(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for standards matching a query."""
        self.load()

        query_lower = query.lower()
        results = []

        categories = [category.lower().rstrip("s") + "s"] if category else list(self._cache.keys())

        for cat in categories:
            if cat not in self._cache:
                continue

            for spec_id, spec in self._cache[cat].items():
                # Match against ID
                if query_lower in spec_id.lower():
                    results.append({"category": cat, "id": spec_id, "spec": spec})
                    continue

                # Match against description
                if "description" in spec and query_lower in spec["description"].lower():
                    results.append({"category": cat, "id": spec_id, "spec": spec})
                    continue

                # Match against use_cases
                if "use_cases" in spec:
                    for use_case in spec["use_cases"]:
                        if query_lower in use_case.lower():
                            results.append({"category": cat, "id": spec_id, "spec": spec})
                            break

        return results

    def get_gear(self, gear_id: str) -> Optional[Dict[str, Any]]:
        """Get a gear specification with calculated dimensions."""
        return self.get_standard("gear", gear_id)

    def list_gears(self) -> List[str]:
        """List all gear IDs."""
        result = self.list_standards("gear")
        return result.get("gears", [])

    def get_formulas(self, category: str) -> Optional[Dict[str, Any]]:
        """Get formulas for a category."""
        self.load()
        return self._formulas.get(category)


# Global instance
_standards_library: Optional[LocalStandardsLibrary] = None


def get_standards_library() -> LocalStandardsLibrary:
    """Get or create the global standards library instance."""
    global _standards_library
    if _standards_library is None:
        _standards_library = LocalStandardsLibrary()
    return _standards_library
