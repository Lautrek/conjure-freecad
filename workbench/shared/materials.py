"""
Shared materials client for Conjure.

Provides unified access to the server's engineering material library
across all client adapters (Blender, FreeCAD, Fusion 360, etc.).
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class EngineeringMaterial:
    """
    Engineering material with physical properties.

    This represents a material from the server's library with
    mechanical, thermal, and other engineering properties.
    """

    id: str
    name: str
    category: str
    description: Optional[str] = None

    # Mechanical properties
    density_kg_m3: Optional[float] = None
    youngs_modulus_pa: Optional[float] = None
    poissons_ratio: Optional[float] = None
    yield_strength_pa: Optional[float] = None
    ultimate_strength_pa: Optional[float] = None
    shear_modulus_pa: Optional[float] = None

    # Thermal properties
    thermal_conductivity_w_mk: Optional[float] = None
    specific_heat_j_kgk: Optional[float] = None
    thermal_expansion_1_k: Optional[float] = None
    melting_point_c: Optional[float] = None

    # Visual properties (for rendering in 3D apps)
    base_color: Optional[tuple] = None  # RGB 0-1
    metallic: Optional[float] = None
    roughness: Optional[float] = None

    source: Optional[str] = None

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "EngineeringMaterial":
        """Create from API response."""
        props = data.get("properties", {})

        # Infer visual properties from material category and properties
        visual = cls._infer_visual_properties(data.get("category", ""), props)

        return cls(
            id=data["id"],
            name=data["name"],
            category=data.get("category", "unknown"),
            description=data.get("description"),
            density_kg_m3=props.get("density_kg_m3"),
            youngs_modulus_pa=props.get("youngs_modulus_pa"),
            poissons_ratio=props.get("poissons_ratio"),
            yield_strength_pa=props.get("yield_strength_pa"),
            ultimate_strength_pa=props.get("ultimate_strength_pa"),
            shear_modulus_pa=props.get("shear_modulus_pa"),
            thermal_conductivity_w_mk=props.get("thermal_conductivity_w_mk"),
            specific_heat_j_kgk=props.get("specific_heat_j_kgk"),
            thermal_expansion_1_k=props.get("thermal_expansion_1_k"),
            melting_point_c=props.get("melting_point_c"),
            base_color=visual.get("base_color"),
            metallic=visual.get("metallic"),
            roughness=visual.get("roughness"),
            source=data.get("source"),
        )

    @staticmethod
    def _infer_visual_properties(category: str, props: Dict[str, Any]) -> Dict[str, Any]:
        """
        Infer visual/rendering properties from engineering properties.

        Maps material category and properties to approximate visual appearance.
        """
        visual = {}

        # Category-based defaults
        category_visuals = {
            "metal": {
                "base_color": (0.8, 0.8, 0.85),
                "metallic": 1.0,
                "roughness": 0.3,
            },
            "plastic": {
                "base_color": (0.9, 0.9, 0.9),
                "metallic": 0.0,
                "roughness": 0.4,
            },
            "composite": {
                "base_color": (0.15, 0.15, 0.15),
                "metallic": 0.0,
                "roughness": 0.2,
            },
            "ceramic": {
                "base_color": (0.95, 0.95, 0.92),
                "metallic": 0.0,
                "roughness": 0.1,
            },
            "elastomer": {
                "base_color": (0.1, 0.1, 0.1),
                "metallic": 0.0,
                "roughness": 0.8,
            },
            "wood": {
                "base_color": (0.55, 0.35, 0.2),
                "metallic": 0.0,
                "roughness": 0.6,
            },
        }

        visual = category_visuals.get(
            category,
            {
                "base_color": (0.7, 0.7, 0.7),
                "metallic": 0.0,
                "roughness": 0.5,
            },
        ).copy()

        # Refine based on specific material characteristics
        if category == "metal":
            # Aluminum is lighter colored
            density = props.get("density_kg_m3", 0)
            if density and density < 3000:  # Likely aluminum
                visual["base_color"] = (0.9, 0.9, 0.92)
                visual["roughness"] = 0.25
            elif density and density > 7500:  # Steel/iron
                visual["base_color"] = (0.6, 0.6, 0.65)
                visual["roughness"] = 0.35

            # Copper has distinctive color
            conductivity = props.get("thermal_conductivity_w_mk", 0)
            if conductivity and conductivity > 350:  # High conductivity = copper
                visual["base_color"] = (0.95, 0.64, 0.54)
                visual["roughness"] = 0.2

        elif category == "plastic":
            # PLA is often light colored
            # ABS tends to be slightly yellow/cream
            # Carbon-filled plastics are dark
            pass

        return visual

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "mechanical": {
                "density_kg_m3": self.density_kg_m3,
                "youngs_modulus_pa": self.youngs_modulus_pa,
                "poissons_ratio": self.poissons_ratio,
                "yield_strength_pa": self.yield_strength_pa,
                "ultimate_strength_pa": self.ultimate_strength_pa,
                "shear_modulus_pa": self.shear_modulus_pa,
            },
            "thermal": {
                "thermal_conductivity_w_mk": self.thermal_conductivity_w_mk,
                "specific_heat_j_kgk": self.specific_heat_j_kgk,
                "thermal_expansion_1_k": self.thermal_expansion_1_k,
                "melting_point_c": self.melting_point_c,
            },
            "visual": {
                "base_color": self.base_color,
                "metallic": self.metallic,
                "roughness": self.roughness,
            },
            "source": self.source,
        }


@dataclass
class MaterialCache:
    """
    Local cache for materials from the server.

    Reduces API calls by caching the material library locally
    with a configurable TTL.
    """

    materials: Dict[str, EngineeringMaterial] = field(default_factory=dict)
    categories: List[str] = field(default_factory=list)
    last_updated: float = 0
    ttl_seconds: float = 3600  # 1 hour default

    def is_valid(self) -> bool:
        """Check if cache is still valid."""
        if not self.materials:
            return False
        return (time.time() - self.last_updated) < self.ttl_seconds

    def get(self, material_id: str) -> Optional[EngineeringMaterial]:
        """Get a material by ID."""
        return self.materials.get(material_id)

    def get_by_category(self, category: str) -> List[EngineeringMaterial]:
        """Get materials by category."""
        return [m for m in self.materials.values() if m.category == category]

    def search(self, query: str) -> List[EngineeringMaterial]:
        """Search materials by name or description."""
        query = query.lower()
        return [
            m
            for m in self.materials.values()
            if query in m.name.lower() or (m.description and query in m.description.lower())
        ]

    def list_all(self) -> List[EngineeringMaterial]:
        """List all materials."""
        return list(self.materials.values())

    def update(self, materials: List[EngineeringMaterial], categories: List[str]):
        """Update cache with new materials."""
        self.materials = {m.id: m for m in materials}
        self.categories = categories
        self.last_updated = time.time()

    def invalidate(self):
        """Invalidate the cache."""
        self.materials = {}
        self.categories = []
        self.last_updated = 0


class MaterialsClient:
    """
    Client for accessing the Conjure material library.

    Provides a unified interface for all client adapters to fetch
    and use engineering materials from the server.

    Usage:
        client = MaterialsClient("http://localhost:8000")
        materials = client.list_materials()
        aluminum = client.get_material("aluminum_6061_t6")
    """

    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        cache_ttl_seconds: float = 3600,
    ):
        """
        Initialize the materials client.

        Args:
            server_url: Base URL of the Conjure server
            api_key: Optional API key for authentication
            cache_ttl_seconds: Cache TTL in seconds (default 1 hour)
        """
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self._cache = MaterialCache(ttl_seconds=cache_ttl_seconds)
        self._object_materials: Dict[str, str] = {}  # object_name -> material_id

    def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """Make an API request to the server."""
        url = f"{self.server_url}{endpoint}"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(url, headers=headers)

        try:
            with urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            raise ConnectionError(f"API error: {e.code} - {e.reason}")
        except URLError as e:
            raise ConnectionError(f"Connection failed: {e.reason}")
        except Exception as e:
            raise ConnectionError(f"Request failed: {str(e)}")

    def refresh_cache(self) -> bool:
        """
        Refresh the material cache from the server.

        Returns:
            True if successful, False otherwise
        """
        try:
            data = self._make_request("/api/v1/simulation/materials")

            materials = [EngineeringMaterial.from_api_response(m) for m in data.get("materials", [])]
            categories = data.get("categories", [])

            self._cache.update(materials, categories)
            return True

        except Exception as e:
            print(f"[Conjure] Failed to refresh material cache: {e}")
            return False

    def _ensure_cache(self):
        """Ensure cache is valid, refresh if needed."""
        if not self._cache.is_valid():
            self.refresh_cache()

    def list_materials(self, category: Optional[str] = None) -> List[EngineeringMaterial]:
        """
        List available materials.

        Args:
            category: Optional category filter

        Returns:
            List of materials
        """
        self._ensure_cache()

        if category:
            return self._cache.get_by_category(category)
        return self._cache.list_all()

    def get_material(self, material_id: str) -> Optional[EngineeringMaterial]:
        """
        Get a specific material by ID.

        Args:
            material_id: Material ID (e.g., "aluminum_6061_t6")

        Returns:
            Material if found, None otherwise
        """
        self._ensure_cache()
        return self._cache.get(material_id)

    def search_materials(self, query: str) -> List[EngineeringMaterial]:
        """
        Search materials by name or description.

        Args:
            query: Search query

        Returns:
            List of matching materials
        """
        self._ensure_cache()
        return self._cache.search(query)

    def get_categories(self) -> List[str]:
        """Get list of material categories."""
        self._ensure_cache()
        return self._cache.categories

    # =========================================================================
    # OBJECT-MATERIAL ASSIGNMENT
    # =========================================================================

    def assign_material(self, object_name: str, material_id: str) -> bool:
        """
        Assign an engineering material to an object.

        Args:
            object_name: Name of the object
            material_id: Material ID to assign

        Returns:
            True if successful
        """
        material = self.get_material(material_id)
        if not material:
            return False

        self._object_materials[object_name] = material_id
        return True

    def get_object_material(self, object_name: str) -> Optional[EngineeringMaterial]:
        """
        Get the engineering material assigned to an object.

        Args:
            object_name: Name of the object

        Returns:
            Material if assigned, None otherwise
        """
        material_id = self._object_materials.get(object_name)
        if not material_id:
            return None
        return self.get_material(material_id)

    def get_object_material_id(self, object_name: str) -> Optional[str]:
        """Get the material ID assigned to an object."""
        return self._object_materials.get(object_name)

    def clear_object_material(self, object_name: str):
        """Clear the material assignment for an object."""
        if object_name in self._object_materials:
            del self._object_materials[object_name]

    def get_all_assignments(self) -> Dict[str, str]:
        """Get all object-material assignments."""
        return self._object_materials.copy()

    def set_assignments(self, assignments: Dict[str, str]):
        """Set object-material assignments (e.g., when loading a file)."""
        self._object_materials = assignments.copy()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_material_for_simulation(
        self,
        object_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get material properties formatted for simulation requests.

        Args:
            object_name: Name of the object

        Returns:
            Dict with material properties for simulation API
        """
        material = self.get_object_material(object_name)
        if not material:
            return None

        return {
            "material_id": material.id,
            "density_kg_m3": material.density_kg_m3,
            "youngs_modulus_pa": material.youngs_modulus_pa,
            "poissons_ratio": material.poissons_ratio,
            "yield_strength_pa": material.yield_strength_pa,
            "thermal_conductivity_w_mk": material.thermal_conductivity_w_mk,
            "specific_heat_j_kgk": material.specific_heat_j_kgk,
        }

    def format_material_display(self, material: EngineeringMaterial) -> str:
        """
        Format material for display in UI.

        Args:
            material: Material to format

        Returns:
            Formatted string for display
        """
        parts = [material.name]

        if material.density_kg_m3:
            parts.append(f"ρ={material.density_kg_m3:.0f} kg/m³")

        if material.youngs_modulus_pa:
            e_gpa = material.youngs_modulus_pa / 1e9
            parts.append(f"E={e_gpa:.1f} GPa")

        return " | ".join(parts)
