import os

from tace.utils.env import acceleration_enabled, enable_acceleration


def test_enable_acceleration_sets_requested_environment(monkeypatch):
    env_names = (
        "TACE_USE_OEQ",
        "TACE_USE_CUE",
        "TACE_USE_EQT",
        "TACE_USE_COMPILE",
        "TACE_USE_TRITON",
    )
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    enable_acceleration(
        enable_oeq=True,
        enable_cue=True,
        enable_eqt=True,
        enable_compile=True,
        enable_triton=True,
    )

    assert {name: os.environ.get(name) for name in env_names} == {
        name: "1" for name in env_names
    }


def test_enable_acceleration_preserves_unrequested_options(monkeypatch):
    monkeypatch.setenv("TACE_USE_OEQ", "1")
    monkeypatch.setenv("TACE_USE_TRITON", "1")

    enable_acceleration()

    assert os.environ["TACE_USE_OEQ"] == "1"
    assert os.environ["TACE_USE_TRITON"] == "1"
    assert acceleration_enabled("oeq")
    assert acceleration_enabled("triton")


def test_enable_acceleration_force_overrides_all_options(monkeypatch):
    for name in (
        "TACE_USE_OEQ",
        "TACE_USE_CUE",
        "TACE_USE_EQT",
        "TACE_USE_COMPILE",
        "TACE_USE_TRITON",
    ):
        monkeypatch.setenv(name, "1")

    enable_acceleration(enable_eqt=True, force=True)

    assert os.environ["TACE_USE_OEQ"] == "0"
    assert os.environ["TACE_USE_CUE"] == "0"
    assert os.environ["TACE_USE_EQT"] == "1"
    assert os.environ["TACE_USE_COMPILE"] == "0"
    assert os.environ["TACE_USE_TRITON"] == "0"
