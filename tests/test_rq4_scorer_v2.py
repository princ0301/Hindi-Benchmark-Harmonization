from src.benchmarks.rq4_scorer_v2 import evaluate_rq4


def test_bare_mcq():
    assert evaluate_rq4("B", "B").status == "correct"
    assert evaluate_rq4("B.", "B").status == "correct"
    assert evaluate_rq4("A", "B").status == "incorrect"


def test_hindi_bare_mcq():
    assert evaluate_rq4("बी.", "B").status == "correct"
    assert evaluate_rq4("ए", "B").status == "incorrect"


def test_explicit_marker():
    assert evaluate_rq4("The correct answer is B.", "B").status == "correct"
    assert evaluate_rq4("Answer: C", "B").status == "incorrect"


def test_unparseable_answer_marker():
    output = """Let's solve it.\n5 - 2 = 3.\n\nAnswer:"""
    result = evaluate_rq4(output, "B")
    assert result.status == "unparseable"
    assert result.predicted is None


def test_do_not_infer_from_arbitrary_letters():
    output = "The options are A, B, C, and D. I think the answer is unclear."
    assert evaluate_rq4(output, "B").status == "unparseable"


def test_yes_no():
    assert evaluate_rq4("YES", "YES").status == "correct"
    assert evaluate_rq4("NO", "YES").status == "incorrect"
    assert evaluate_rq4("हाँ", "YES").status == "correct"


def test_reasoning_with_explicit_answer_is_parseable():
    output = "5 - 2 = 3. Therefore, the correct answer is B."
    result = evaluate_rq4(output, "B")
    assert result.status == "correct"
    assert result.predicted == "b"
