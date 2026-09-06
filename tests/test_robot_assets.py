"""Resolution and provenance of the third-party robot assets.

The benchmark does not vendor robot models, so these tests must pass on a
machine that has none: everything that needs the asset skips, and everything
about *declaring* the asset is checked unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multisport_sim.benchmark import assets


class TestAssetSource:
    def test_the_panda_declares_complete_provenance(self) -> None:
        payload = assets.FRANKA_PANDA.to_dict()
        assert payload["license"] == "Apache-2.0"
        assert payload["upstream_url"].startswith("https://")
        assert payload["version"]
        # Every local change must be written down; an undocumented modification
        # is the failure mode this record exists to prevent.
        assert payload["modifications"]

    def test_an_incomplete_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="license_id"):
            assets.AssetSource(
                asset_id="x",
                relative_path="x/x.xml",
                upstream_url="https://example.invalid",
                version="1",
                license_id="  ",
                license_path="x/LICENSE",
            )

    def test_the_manifest_lists_every_asset_with_availability(self) -> None:
        manifest = assets.license_manifest()
        assert {row["asset_id"] for row in manifest} == set(assets.ASSETS)
        assert all(isinstance(row["available"], bool) for row in manifest)


class TestResolution:
    def test_an_environment_variable_takes_priority(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = tmp_path / "some_robot" / "robot.xml"
        model.parent.mkdir()
        model.write_text("<mujoco/>", encoding="utf-8")
        monkeypatch.setenv("MULTISPORT_MENAGERIE_PATH", str(tmp_path))
        assert assets.menagerie_root() == tmp_path

    def test_an_empty_directory_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale variable must not shadow a working checkout."""
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setenv("MULTISPORT_MENAGERIE_PATH", str(empty))
        assert assets.menagerie_root() != empty

    def test_a_missing_checkout_resolves_to_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        for variable in assets.MENAGERIE_ENV_VARS:
            monkeypatch.setenv(variable, str(empty))
        monkeypatch.setattr(assets, "MENAGERIE_DEFAULT", empty)
        assert assets.menagerie_root() is None
        assert assets.resolve_asset(assets.FRANKA_PANDA) is None
        assert not assets.asset_available(assets.FRANKA_PANDA)

    def test_requiring_a_missing_asset_explains_how_to_get_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        for variable in assets.MENAGERIE_ENV_VARS:
            monkeypatch.setenv(variable, str(empty))
        monkeypatch.setattr(assets, "MENAGERIE_DEFAULT", empty)
        with pytest.raises(assets.AssetUnavailableError) as error:
            assets.require_asset(assets.FRANKA_PANDA)
        message = str(error.value)
        assert "sparse-checkout" in message
        assert "MULTISPORT_MENAGERIE_PATH" in message


@pytest.mark.skipif(
    not assets.asset_available(assets.FRANKA_PANDA),
    reason="the Franka Panda asset is not installed; see docs/ROBOT_LAYER.md",
)
class TestInstalledAsset:
    def test_the_declared_model_file_exists(self) -> None:
        path = assets.require_asset(assets.FRANKA_PANDA)
        assert path.is_file()
        assert path.name == "panda_nohand.xml"

    def test_the_declared_licence_is_readable_and_matches(self) -> None:
        text = assets.license_text(assets.FRANKA_PANDA)
        assert "Apache License" in text
