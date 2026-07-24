import os

from tace.utils.env import enable_acceleration


def test_enable_acceleration_sets_requested_environment(monkeypatch):
    env_names = (
        "TACE_USE_OEQ",
        "TACE_USE_CUE",
        "TACE_USE_EQT",
        "TACE_USE_COMPILE",
    )
    for name in env_names:
        monkeypatch.delenv(name, raising=False)

    enable_acceleration(
        enable_oeq=True,
        enable_cue=True,
        enable_eqt=True,
        enable_compile=True,
    )

    assert {name: os.environ.get(name) for name in env_names} == {
        name: "1" for name in env_names
    }


def test_enable_acceleration_preserves_external_environment(monkeypatch):
    monkeypatch.setenv("TACE_USE_OEQ", "1")

    enable_acceleration()

    assert os.environ["TACE_USE_OEQ"] == "1"
