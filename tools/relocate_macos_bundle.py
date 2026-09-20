#!/usr/bin/env python3
"""Copy and relocate non-system dylib dependencies into a macOS app bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


MACHO_MAGICS = {
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}


def run(*arguments: str) -> str:
    completed = subprocess.run(
        arguments,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}\n{detail}"
        )
    return completed.stdout


def is_macho(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(4) in MACHO_MAGICS
    except OSError:
        return False


def dependencies(path: Path) -> list[str]:
    lines = run("/usr/bin/otool", "-L", str(path)).splitlines()[1:]
    return [line.strip().split(" (", 1)[0] for line in lines if line.strip()]


def install_id(path: Path) -> str | None:
    lines = run("/usr/bin/otool", "-D", str(path)).splitlines()[1:]
    return lines[0].strip() if lines else None


def rpaths(path: Path) -> set[str]:
    lines = run("/usr/bin/otool", "-l", str(path)).splitlines()
    result: set[str] = set()
    for index, line in enumerate(lines[:-1]):
        if line.strip() == "cmd LC_RPATH":
            for candidate in lines[index + 1 : index + 5]:
                value = candidate.strip()
                if value.startswith("path "):
                    result.add(value.split()[1])
                    break
    return result


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def app_macho_files(app: Path) -> list[Path]:
    return [path for path in app.rglob("*") if is_macho(path)]


def copy_dependency(source: Path, destination_directory: Path) -> Path:
    source = source.resolve()
    destination = destination_directory / source.name
    if destination.exists():
        if digest(source) != digest(destination):
            raise RuntimeError(
                f"dylib basename collision for {source.name}: {source} and {destination}"
            )
        return destination
    shutil.copy2(source, destination)
    os.chmod(destination, source.stat().st_mode)
    return destination


def collect_dependencies(app: Path, brew_prefix: Path, libraries: Path) -> None:
    queue = app_macho_files(app)
    visited: set[Path] = set()
    while queue:
        binary = queue.pop()
        resolved = binary.resolve()
        if resolved in visited:
            continue
        visited.add(resolved)
        for dependency in dependencies(binary):
            dependency_path = Path(dependency)
            if not dependency.startswith(str(brew_prefix) + "/"):
                continue
            if not dependency_path.exists():
                raise RuntimeError(f"missing Homebrew dependency: {dependency}")
            copied = copy_dependency(dependency_path, libraries)
            if copied.resolve() not in visited:
                queue.append(copied)


def relocate(app: Path, brew_prefix: Path, libraries: Path) -> None:
    for binary in app_macho_files(app):
        replacements: list[tuple[str, str]] = []
        binary_install_id = install_id(binary)
        for dependency in dependencies(binary):
            if dependency.startswith(str(brew_prefix) + "/"):
                if dependency == binary_install_id:
                    continue
                bundled = libraries / Path(dependency).resolve().name
                if not bundled.exists():
                    raise RuntimeError(f"dependency was not collected: {dependency}")
                replacements.append((dependency, f"@rpath/{bundled.name}"))
        if not replacements:
            if not (binary_install_id and binary_install_id.startswith(str(brew_prefix) + "/")):
                continue
        else:
            for old, new in replacements:
                run("/usr/bin/install_name_tool", "-change", old, new, str(binary))
            relative_libraries = os.path.relpath(libraries, binary.parent)
            loader_rpath = (
                "@loader_path"
                if relative_libraries == "."
                else f"@loader_path/{relative_libraries}"
            )
            if loader_rpath not in rpaths(binary):
                run("/usr/bin/install_name_tool", "-add_rpath", loader_rpath, str(binary))
        if binary_install_id and binary_install_id.startswith(str(brew_prefix) + "/"):
            run("/usr/bin/install_name_tool", "-id", f"@rpath/{binary.name}", str(binary))
        if binary.parent == libraries and binary.suffix in {".dylib", ".so", ""}:
            run("/usr/bin/install_name_tool", "-id", f"@rpath/{binary.name}", str(binary))


def verify(app: Path, brew_prefix: Path) -> None:
    unresolved: list[str] = []
    for binary in app_macho_files(app):
        for dependency in dependencies(binary):
            if dependency.startswith(str(brew_prefix) + "/"):
                unresolved.append(f"{binary}: {dependency}")
    if unresolved:
        raise RuntimeError("unrelocated dependencies:\n" + "\n".join(unresolved))


def sign_macho_files(app: Path, identity: str) -> None:
    binaries = sorted(
        app_macho_files(app),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for binary in binaries:
        run(
            "/usr/bin/codesign",
            "--force",
            "--sign",
            identity,
            "--timestamp=none",
            str(binary),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--brew-prefix", type=Path, default=Path("/opt/homebrew"))
    parser.add_argument("--sign-identity")
    arguments = parser.parse_args()
    app = arguments.app.resolve()
    libraries = app / "Contents/Frameworks/RuntimeLibraries"
    libraries.mkdir(parents=True, exist_ok=True)
    collect_dependencies(app, arguments.brew_prefix.resolve(), libraries)
    relocate(app, arguments.brew_prefix.resolve(), libraries)
    verify(app, arguments.brew_prefix.resolve())
    if arguments.sign_identity:
        sign_macho_files(app, arguments.sign_identity)
    print(f"Relocated {len(list(libraries.iterdir()))} Homebrew runtime libraries")


if __name__ == "__main__":
    main()
