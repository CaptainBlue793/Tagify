from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_example_analysis_survives_setting_change():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Tagify.py")).run()
    assert not app.exception
    app.button[0].click().run()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["analysis"]["industries"]
    app.slider[0].set_value(3).run()
    assert app.session_state["analysis"]["topics"]
    assert not app.exception


def test_empty_analysis_reports_validation():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Tagify.py")).run()
    app.button[1].click().run()
    assert app.error
    assert not app.exception


def test_text_upload_takes_precedence_over_pasted_text():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "Tagify.py")).run()
    if not hasattr(app.file_uploader[0], "set_value"):
        pytest.skip("Upload testing requires a recent Streamlit release")
    app.text_area(key="source").set_value("gardens flowers").run()
    app.file_uploader[0].set_value(("article.txt", b"Software engineering and cybersecurity protect databases.", "text/plain")).run()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["analysis"]["source"].startswith("Software")
