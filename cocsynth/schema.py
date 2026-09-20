"""Ground-truth label schema.

These models are the contract between the generator and anything that consumes the
dataset. The JSON they serialise to is what a training loop diffs predictions against,
so every field here is something a model is expected to get right.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"

#: Grid headings in clockwise order; index is the `direction` value.
HEADINGS: tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


#: Grid-space unit vector per heading, in the same clockwise order as HEADINGS.
#: Grid north is -y, east is +x; the renderer projects these into screen space.
HEADING_VECTORS: tuple[tuple[int, int], ...] = (
    (0, -1),   # N
    (1, -1),   # NE
    (1, 0),    # E
    (1, 1),    # SE
    (0, 1),    # S
    (-1, 1),   # SW
    (-1, 0),   # W
    (-1, -1),  # NW
)


def heading_vector(direction: int, directions: int) -> tuple[int, int]:
    """Grid-space direction a building faces. Fixed buildings face grid north."""
    if directions <= 1:
        return HEADING_VECTORS[0]
    step = len(HEADINGS) // directions
    return HEADING_VECTORS[(direction * step) % len(HEADINGS)]


def heading_of(direction: int, directions: int) -> str:
    """Human-readable heading for a direction index.

    A building with a single facing is always reported as "N" so consumers never
    have to special-case fixed buildings.
    """
    if directions <= 1:
        return HEADINGS[0]
    step = len(HEADINGS) // directions
    return HEADINGS[(direction * step) % len(HEADINGS)]


def rotation_deg_of(direction: int, directions: int) -> int:
    """Clockwise degrees from north for a direction index."""
    if directions <= 1:
        return 0
    return int(round(direction * 360 / directions)) % 360


class GridMeta(BaseModel):
    """Everything needed to map tile coordinates to pixels and back."""

    tiles: int = Field(gt=0, description="Playable grid is square: tiles x tiles")
    tile_w: int = Field(gt=0, description="Diamond width of one tile in pixels")
    tile_h: int = Field(gt=0, description="Diamond height of one tile in pixels")
    projection: Literal["isometric"] = "isometric"
    origin_px: tuple[int, int] = Field(description="Pixel position of tile (0, 0)'s top vertex")


class BuildingLabel(BaseModel):
    """One placed building instance."""

    id: int = Field(ge=0, description="Unique within this image; matches the id mask")
    type: str
    level: int = Field(ge=1)
    tile: tuple[int, int] = Field(description="(x, y) of the footprint's top-left tile")
    footprint: tuple[int, int] = Field(description="(width, height) in tiles")

    direction: int = Field(default=0, ge=0, description="Facing index; 0 for fixed buildings")
    heading: str = Field(default="N", description="Compass form of `direction`")
    rotation_deg: int = Field(default=0, ge=0, lt=360)

    bbox_px: tuple[int, int, int, int] = Field(description="(x, y, w, h) tight over visible alpha")
    footprint_polygon_px: list[tuple[int, int]] = Field(
        description="The 4 corners of the footprint diamond, in pixels"
    )
    visibility: float = Field(gt=0.0, le=1.0, description="Fraction of the sprite not occluded")
    sprite: str = Field(description="Sprite file actually used, after fallback resolution")

    @field_validator("heading")
    @classmethod
    def _known_heading(cls, v: str) -> str:
        if v not in HEADINGS:
            raise ValueError(f"heading must be one of {HEADINGS}, got {v!r}")
        return v

    @field_validator("footprint_polygon_px")
    @classmethod
    def _quad(cls, v: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(v) != 4:
            raise ValueError(f"footprint polygon must have 4 corners, got {len(v)}")
        return v


class BaseLabel(BaseModel):
    """The complete ground truth for one rendered base image."""

    schema_version: str = SCHEMA_VERSION
    image: str
    image_size: tuple[int, int]
    town_hall_level: int = Field(ge=1)
    seed: int
    grid: GridMeta
    buildings: list[BuildingLabel]

    def by_type(self) -> dict[str, list[BuildingLabel]]:
        """Group instances by building type."""
        out: dict[str, list[BuildingLabel]] = {}
        for b in self.buildings:
            out.setdefault(b.type, []).append(b)
        return out

    def counts(self) -> dict[str, int]:
        """Instances per building type -- what the cap tests assert against."""
        return {k: len(v) for k, v in self.by_type().items()}
