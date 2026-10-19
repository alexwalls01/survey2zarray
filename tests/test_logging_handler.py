import logging

from survey2zarray.utils.logging import ProgressAwareHandler, configure


class TestProgressAwareHandler:
    """Verify log records preserve tqdm display state."""

    def record(self, message="hello"):
        return logging.LogRecord(
            "survey2zarray.test", logging.INFO, __file__, 1, message, None, None
        )

    def test_records_go_through_tqdm_write(self, monkeypatch):
        import tqdm as tqdm_module

        written = []
        monkeypatch.setattr(
            tqdm_module.tqdm,
            "write",
            classmethod(lambda cls, text, **kwargs: written.append(text)),
        )
        handler = ProgressAwareHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.emit(self.record("mid-extraction"))

        assert written == ["mid-extraction"]

    def test_it_still_writes_where_tqdm_is_absent(self, monkeypatch, capsys):
        import builtins

        real_import = builtins.__import__

        def without_tqdm(name, *args, **kwargs):
            if name == "tqdm":
                raise ImportError("no tqdm")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_tqdm)
        handler = ProgressAwareHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.emit(self.record("still reported"))

        assert "still reported" in capsys.readouterr().err

    def test_configure_installs_it(self):
        root = configure(logging.INFO)
        handlers = [h for h in root.handlers if getattr(h, "_survey2zarray", False)]
        assert handlers and all(isinstance(h, ProgressAwareHandler) for h in handlers)
