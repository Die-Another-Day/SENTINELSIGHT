import os
from pathlib import Path
from typing import Dict, List

from loguru import logger


class DirectoryBrowser:
    """
    SentinelSight Directory Browser

    Provides safe directory traversal and metadata
    for frontend file selection.
    """

    # =====================================================
    # Directory Listing
    # =====================================================
    def list_directory(self, path: str = None) -> Dict:
        """
        List directory contents with metadata
        """
        try:
            if not path:
                path = str(Path.home())

            path_obj = Path(path)

            if not path_obj.exists():
                return {
                    "error": "Path does not exist",
                    "path": path,
                    "items": []
                }

            if not path_obj.is_dir():
                path_obj = path_obj.parent

            items: List[Dict] = []

            try:
                for item in sorted(
                    path_obj.iterdir(),
                    key=lambda x: (not x.is_dir(), x.name.lower())
                ):
                    try:
                        is_directory = item.is_dir()

                        item_info = {
                            "name": item.name,
                            "path": str(item),
                            "is_directory": is_directory,
                            "is_hidden": item.name.startswith(".")
                        }

                        if not is_directory:
                            try:
                                stat = item.stat()
                                item_info.update({
                                    "size": stat.st_size,
                                    "extension": item.suffix.lower()
                                })
                            except Exception:
                                pass

                        items.append(item_info)

                    except PermissionError:
                        continue

            except PermissionError:
                return {
                    "error": "Permission denied",
                    "path": str(path_obj),
                    "items": []
                }

            parent_path = (
                str(path_obj.parent)
                if path_obj.parent != path_obj
                else None
            )

            return {
                "path": str(path_obj),
                "parent": parent_path,
                "items": items,
                "count": len(items)
            }

        except Exception as exc:
            logger.error(f"[SentinelSight] Directory browse failed: {exc}")
            return {
                "error": str(exc),
                "path": path,
                "items": []
            }

    # =====================================================
    # Common Locations
    # =====================================================
    def get_common_locations(self) -> List[Dict]:
        """
        Return commonly used directories and drives
        """
        locations: List[Dict] = []
        home = Path.home()

        common_dirs = [
            ("Home", home),
            ("Desktop", home / "Desktop"),
            ("Documents", home / "Documents"),
            ("Downloads", home / "Downloads"),
            ("Pictures", home / "Pictures"),
        ]

        for name, path in common_dirs:
            if path.exists():
                locations.append({
                    "name": name,
                    "path": str(path),
                    "icon": name.lower()
                })

        # macOS / Linux volumes
        volumes = Path("/Volumes")
        if volumes.exists():
            for volume in volumes.iterdir():
                if volume.is_dir() and volume.name != "Macintosh HD":
                    locations.append({
                        "name": volume.name,
                        "path": str(volume),
                        "icon": "drive"
                    })

        # Windows drives
        if os.name == "nt":
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    locations.append({
                        "name": f"Drive {letter}:",
                        "path": drive,
                        "icon": "drive"
                    })

        return locations
