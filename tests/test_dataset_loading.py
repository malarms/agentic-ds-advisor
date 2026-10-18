import pandas as pd

from app.agents.run_advisor import load_tabular_dataset


def test_load_comma_separated_csv(tmp_path):
    path = tmp_path / "comma.csv"
    path.write_text(
        "age,job,y\n"
        '30,"blue-collar",no\n'
        '45,"management",yes\n',
        encoding="utf-8",
    )

    df = load_tabular_dataset(str(path))

    assert df.shape == (2, 3)
    assert df.columns.tolist() == ["age", "job", "y"]
    assert df["age"].tolist() == [30, 45]
    assert df["y"].tolist() == ["no", "yes"]


def test_load_semicolon_separated_csv(tmp_path):
    path = tmp_path / "semicolon.csv"
    path.write_text(
        'age;"job";"y"\n'
        '30;"blue-collar";"no"\n'
        '45;"management";"yes"\n',
        encoding="utf-8",
    )

    df = load_tabular_dataset(str(path))

    assert df.shape == (2, 3)
    assert df.columns.tolist() == ["age", "job", "y"]
    assert df["age"].tolist() == [30, 45]
    assert df["y"].tolist() == ["no", "yes"]


def test_load_real_bank_marketing_dataset():
    path = "data/bank+marketing/bank/bank-full.csv"

    df = load_tabular_dataset(path)

    assert df.shape == (45211, 17)
    assert df.columns[0] == "age"
    assert df.columns[-1] == "y"
    assert pd.api.types.is_numeric_dtype(df["age"])
